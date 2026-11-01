// ECGops web page: talks to the FastAPI service in src/ecgops/api.py.
// Plain JavaScript, no build step and no external libraries, so it works offline.
//
// The server returns P(abnormal) for every beat once. Everything that depends on the
// threshold (labels, badges, counts, accuracy, confusion matrix, chart colours) is
// recomputed here in the browser whenever the slider moves; nothing is uploaded again.
"use strict";

const POLL_INTERVAL_MS = 300;
const THEME_KEY = "ecgops-theme";
const NAMES = ["normal", "abnormal"];
const TITLES = ["Normal", "Abnormal"];

const state = {
  samplingRate: 125,      // Hz, from /health
  threshold: 0.5,         // current slider value
  defaultThreshold: 0.5,  // the model's default threshold, from /health
  samples: [],
  sample: null,           // { beat, probability } for the selected sample
  sampleRun: 0,           // bumped when the sample changes; late /predict answers are ignored
  job: null,              // { id, filename, hasLabels, threshold, rows, beatCache }
  selectedBeat: null,     // 1-based beat number drawn in the results chart
  selectedRow: null,      // its <tr>
  run: 0,                 // bumped by clearResults(); stale async work compares and gives up
  xhr: null,              // in-flight upload
  pollTimer: null,        // pending setTimeout of the job poll
  cancelPoll: null,       // settles the current poll promise when results are cleared
  pendingJobId: null,     // job created on the server but not loaded yet
  frame: 0,               // requestAnimationFrame id for slider updates
};

const $ = (id) => document.getElementById(id);
const labelOf = (p) => (p >= state.threshold ? 1 : 0);
const pct = (part, whole) => (whole ? `${((100 * part) / whole).toFixed(1)}%` : "–");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

/** Read a design token (CSS custom property) from the current theme. */
function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---------------------------------------------------------------- helpers

/** Turn an error response body into a readable message (FastAPI uses {"detail": ...}). */
function errorMessage(body, status) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("\n");
  }
  return `Request failed (HTTP ${status}).`;
}

async function getJSON(url, options) {
  const response = await fetch(url, options);
  let body = null;
  try { body = await response.json(); } catch (_) { /* not JSON */ }
  if (!response.ok) throw new Error(errorMessage(body, response.status));
  return body;
}

/** A "Normal" / "Abnormal" badge: colour plus text plus shape (circle / triangle). */
function pill(label) {
  const span = document.createElement("span");
  span.className = `pill ${NAMES[label]}`;
  span.textContent = TITLES[label];
  return span;
}

function verdict(ok) {
  const span = document.createElement("span");
  span.className = `verdict ${ok ? "yes" : "no"}`;
  span.textContent = ok ? "✓ Yes" : "✗ No";
  return span;
}

// ---------------------------------------------------------------- theme

function storedTheme() {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch (_) {
    return "system"; // storage blocked (private mode, strict settings): follow the OS
  }
}

function applyTheme(choice, save) {
  const root = document.documentElement;
  if (choice === "light" || choice === "dark") root.setAttribute("data-theme", choice);
  else root.removeAttribute("data-theme");
  if (save) {
    try {
      if (choice === "system") localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, choice);
    } catch (_) { /* not remembered, but the page still switches */ }
  }
  for (const input of document.querySelectorAll('#theme-toggle input[name="theme"]')) {
    input.checked = input.value === choice;
    input.parentElement.classList.toggle("checked", input.checked);
  }
  redrawCharts();
}

function setupTheme() {
  applyTheme(storedTheme(), false);
  $("theme-toggle").addEventListener("change", (e) => {
    if (e.target.name === "theme") applyTheme(e.target.value, true);
  });
  // "System" follows live OS changes; the canvas has to be repainted with the new tokens.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawCharts);
}

// ---------------------------------------------------------------- chart

/**
 * Draw one beat on a canvas, sharp on high-DPI screens: light grid, time axis in seconds,
 * amplitude 0..1, line coloured by the predicted label. values = null draws an empty chart
 * with a message instead.
 */
function drawWave(canvas, values, label, title, emptyText) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return; // hidden
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const font = token("--font") || "system-ui, sans-serif";
  const narrow = width < 420;
  const pad = { left: 58, right: 14, top: title ? 34 : 14, bottom: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const n = values ? values.length : 187;
  const seconds = (n - 1) / state.samplingRate;
  const x = (i) => pad.left + (i / (n - 1)) * plotW;
  const y = (v) => pad.top + (1 - Math.min(Math.max(v, 0), 1)) * plotH;

  // Grid + tick labels.
  ctx.lineWidth = 1;
  ctx.font = `11px ${font}`;
  ctx.fillStyle = token("--chart-text");
  ctx.strokeStyle = token("--chart-grid");
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (let v = 0; v <= 1.0001; v += 0.25) {
    const py = Math.round(y(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(width - pad.right, py); ctx.stroke();
    ctx.fillText(v.toFixed(2), pad.left - 8, py);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const step = narrow ? 0.5 : 0.25;
  for (let t = 0; t <= seconds + 1e-9; t += step) {
    const px = Math.round(pad.left + (t / seconds) * plotW) + 0.5;
    ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, pad.top + plotH); ctx.stroke();
    ctx.fillText(t.toFixed(2), px, pad.top + plotH + 6);
  }

  // Axes + axis titles.
  ctx.strokeStyle = token("--chart-axis");
  ctx.beginPath();
  ctx.moveTo(pad.left + 0.5, pad.top);
  ctx.lineTo(pad.left + 0.5, pad.top + plotH + 0.5);
  ctx.lineTo(width - pad.right, pad.top + plotH + 0.5);
  ctx.stroke();
  ctx.font = `600 11px ${font}`;
  ctx.fillText("Time (s)", pad.left + plotW / 2, height - 16);
  ctx.save();
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "middle";
  ctx.fillText("Amplitude", 0, 0);
  ctx.restore();

  if (!values) {
    ctx.fillStyle = token("--chart-empty");
    ctx.font = `13px ${font}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(emptyText || "Nothing to draw yet", pad.left + plotW / 2, pad.top + plotH / 2);
    return;
  }

  // Waveform: soft area fill plus the line, both in the prediction's colour.
  const colour = token(label === 1 ? "--abnormal" : "--normal");
  ctx.beginPath();
  values.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
  ctx.lineTo(x(n - 1), y(0));
  ctx.lineTo(x(0), y(0));
  ctx.closePath();
  ctx.globalAlpha = 0.1;
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.globalAlpha = 1;

  ctx.beginPath();
  values.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
  ctx.strokeStyle = colour;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();

  if (title) {
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.font = `600 13px ${font}`;
    ctx.fillStyle = colour;
    ctx.fillText(label === 1 ? "▲" : "●", pad.left, 20);
    ctx.fillStyle = token("--text");
    ctx.fillText(fitText(ctx, title, width - pad.left - 16 - pad.right), pad.left + 16, 20);
  }
}

/** Shorten text with an ellipsis so it fits in maxWidth pixels (narrow phone charts). */
function fitText(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let end = text.length;
  while (end > 1 && ctx.measureText(`${text.slice(0, end)}…`).width > maxWidth) end -= 1;
  return `${text.slice(0, end).trimEnd()}…`;
}

function redrawCharts() {
  renderSample();
  drawSelectedBeat();
}

// ---------------------------------------------------------------- health + threshold

function setModelStatus(text, kind) {
  const el = $("model-status");
  el.dataset.state = kind;
  el.querySelector(".status-text").textContent = text;
  el.title = text;
}

async function loadHealth() {
  try {
    const health = await getJSON("/health");
    state.samplingRate = health.sampling_rate_hz || state.samplingRate;
    state.defaultThreshold = state.threshold = health.threshold;
    $("threshold").value = String(health.threshold);
    updateThresholdUI();
    if (health.model_loaded) setModelStatus(`Model ready: ${health.model}`, "ok");
    else setModelStatus(`Model not loaded: ${health.error}`, "bad");
  } catch (err) {
    setModelStatus(`API not reachable: ${err.message}`, "bad");
  }
}

/** Slider value, track fill, Reset button and download link (cheap; runs on every input). */
function updateThresholdUI() {
  const slider = $("threshold");
  $("threshold-value").textContent = state.threshold.toFixed(2);
  slider.style.setProperty("--fill", `${state.threshold * 100}%`);
  const reset = $("threshold-reset");
  reset.disabled = Math.abs(state.threshold - state.defaultThreshold) < 1e-9;
  reset.textContent = `Reset to default (${state.defaultThreshold.toFixed(2)})`;
  if (state.job) {
    $("download").href = `/jobs/${state.job.id}/results.csv?threshold=${state.threshold}`;
  }
}

function onThresholdInput() {
  state.threshold = Number($("threshold").value);
  updateThresholdUI();
  // Coalesce fast slider moves into one re-render per frame.
  if (!state.frame) {
    state.frame = requestAnimationFrame(() => {
      state.frame = 0;
      renderSample();
      updateResults();
    });
  }
}

function resetThreshold() {
  $("threshold").value = String(state.defaultThreshold);
  onThresholdInput();
}

// ---------------------------------------------------------------- samples

async function loadSamples() {
  const select = $("sample-select");
  try {
    const data = await getJSON("/samples");
    state.samples = data.beats;
    select.replaceChildren(new Option("Choose a sample…", ""));
    for (const beat of data.beats) {
      const text = `Beat ${beat.id}` + (beat.label_name ? ` (true: ${beat.label_name})` : "");
      select.appendChild(new Option(text, String(beat.id)));
    }
    select.disabled = false;
    if (data.beats.length) {
      select.value = String(data.beats[0].id);
      await selectSample();
    } else {
      renderSample();
    }
  } catch (err) {
    select.replaceChildren(new Option("No samples available", ""));
    renderSample();
    $("sample-result").textContent = err.message;
  }
}

async function selectSample() {
  const run = ++state.sampleRun;
  const beat = state.samples.find((b) => String(b.id) === $("sample-select").value);
  state.sample = beat ? { beat, probability: null, error: null } : null;
  renderSample();
  if (!beat) return;
  try {
    const result = await getJSON("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ beats: [beat.values] }),
    });
    if (run !== state.sampleRun) return; // another sample was chosen meanwhile
    state.sample.probability = result.predictions[0].probability;
  } catch (err) {
    if (run !== state.sampleRun) return;
    state.sample.error = err.message;
  }
  renderSample();
}

function clearSample() {
  state.sampleRun += 1;
  state.sample = null;
  $("sample-select").value = "";
  renderSample();
}

function renderSample() {
  const canvas = $("sample-canvas");
  const out = $("sample-result");
  if (!state.sample) {
    drawWave(canvas, null, 0, "", "Choose a sample beat to draw it");
    out.replaceChildren();
    return;
  }
  const { beat, probability, error } = state.sample;
  if (probability === null) {
    drawWave(canvas, beat.values, 0, `Beat ${beat.id}`);
    out.textContent = error || "Predicting…";
    return;
  }
  const label = labelOf(probability);
  drawWave(canvas, beat.values, label, `Beat ${beat.id}: ${TITLES[label]}`);

  const prob = document.createElement("span");
  prob.className = "meta";
  prob.append("P(abnormal) ");
  const num = document.createElement("span");
  num.className = "num";
  num.textContent = probability.toFixed(4);
  prob.append(num);
  out.replaceChildren("Prediction", pill(label), prob);
  if (beat.label !== null && beat.label !== undefined) {
    const truth = document.createElement("span");
    truth.className = "meta";
    truth.textContent = `True label: ${beat.label_name}`;
    out.append(truth, verdict(beat.label === label));
  }
}

// ---------------------------------------------------------------- upload + progress

/** step 1 = upload, 2 = analysis. aria-live is on the text, the bar carries aria-valuenow. */
function showProgress(text, percent, done = false) {
  const value = Math.max(0, Math.min(100, Math.round(percent)));
  $("progress-area").hidden = false;
  $("progress-area").classList.toggle("done", done);
  $("progress-text").textContent = text;
  $("progress-percent").textContent = `${value}%`;
  $("progress-fill").style.width = `${value}%`;
  $("progress-bar").setAttribute("aria-valuenow", String(value));
}

function hideProgress() {
  $("progress-area").hidden = true;
  $("progress-area").classList.remove("done");
  $("progress-fill").style.width = "0";
  $("progress-bar").setAttribute("aria-valuenow", "0");
}

function showError(message) {
  const el = $("upload-error");
  el.textContent = message;
  el.hidden = !message;
}

function setBusy(busy) {
  $("dropzone").classList.toggle("busy", busy);
  $("dropzone").setAttribute("aria-busy", String(busy));
}

/** Phase 1: upload with real XMLHttpRequest progress events. Resolves with {job_id, total}. */
function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    state.xhr = xhr;
    xhr.open("POST", "/jobs");
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (xhr !== state.xhr || !event.lengthComputable) return;
      const percent = (100 * event.loaded) / event.total;
      showProgress(`Step 1 of 2 · Uploading ${file.name}`, percent);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new Error(errorMessage(xhr.response, xhr.status)));
    };
    xhr.onerror = () => reject(new Error("Network error: is `uv run serve` still running?"));
    xhr.onabort = () => reject(new Error("Upload cancelled."));
    xhr.onloadend = () => { if (state.xhr === xhr) state.xhr = null; };
    xhr.send(form);
  });
}

/**
 * Phase 2: poll GET /jobs/{id} until the background job is done (or failed).
 * Resolves with null if clearResults() cancelled it; late answers of an old run are ignored.
 */
function pollJob(jobId, run) {
  return new Promise((resolve, reject) => {
    state.cancelPoll = () => resolve(null);
    const tick = async () => {
      state.pollTimer = null;
      if (run !== state.run) return resolve(null);
      try {
        const job = await getJSON(`/jobs/${jobId}`);
        if (run !== state.run) return resolve(null);
        if (job.status === "error") return reject(new Error(`Analysis failed: ${job.error}`));
        const done = job.status === "done";
        showProgress(
          done
            ? `Done · ${job.total} beats analysed`
            : `Step 2 of 2 · Analysing ${job.processed} / ${job.total} beats`,
          job.percent, done);
        if (done) return resolve(job);
        state.pollTimer = setTimeout(tick, POLL_INTERVAL_MS);
      } catch (err) {
        if (run !== state.run) return resolve(null);
        reject(err);
      }
    };
    tick();
  });
}

async function handleFile(file) {
  if (!file) return;
  clearResults(); // a new upload replaces the old results (and frees the old job)
  const run = state.run;
  setBusy(true);
  $("clear-results").disabled = false; // lets the user cancel
  showProgress(`Step 1 of 2 · Uploading ${file.name}`, 0);
  try {
    const created = await uploadFile(file);
    if (run !== state.run) { forgetJob(created.job_id); return; }
    state.pendingJobId = created.job_id;
    showProgress(`Step 2 of 2 · Analysing 0 / ${created.total} beats`, 0);
    const job = await pollJob(created.job_id, run);
    if (!job || run !== state.run) return;
    state.pendingJobId = null;
    state.cancelPoll = null;
    loadJob(job);
  } catch (err) {
    if (run !== state.run) return; // cleared meanwhile: stay quiet
    hideProgress();
    showError(err.message);
    if (state.pendingJobId) { forgetJob(state.pendingJobId); state.pendingJobId = null; }
    $("clear-results").disabled = false; // so the error can be cleared
  } finally {
    if (run === state.run) setBusy(false);
  }
}

function setupDropzone() {
  const zone = $("dropzone");
  const input = $("file-input");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => handleFile(input.files[0]));
  let depth = 0; // dragenter/dragleave also fire for child elements
  zone.addEventListener("dragenter", (e) => { e.preventDefault(); depth += 1; zone.classList.add("dragover"); });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); });
  zone.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) zone.classList.remove("dragover"); });
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    zone.classList.remove("dragover");
    handleFile(e.dataTransfer.files[0]);
  });
  // A file dropped next to the zone should not navigate away from the page.
  for (const type of ["dragover", "drop"]) window.addEventListener(type, (e) => e.preventDefault());
}

// ---------------------------------------------------------------- clear

/** Tell the server it can forget a job (fire and forget; it expires anyway). */
function forgetJob(jobId) {
  fetch(`/jobs/${jobId}`, { method: "DELETE" }).catch(() => {});
}

/**
 * Reset the results without reloading: stop the upload / poll, ignore late answers (the run
 * counter changes), forget the job on the server and empty every results element.
 */
function clearResults() {
  state.run += 1;
  if (state.xhr) { const xhr = state.xhr; state.xhr = null; xhr.abort(); }
  if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; }
  if (state.cancelPoll) { state.cancelPoll(); state.cancelPoll = null; }
  if (state.job) forgetJob(state.job.id);
  if (state.pendingJobId) forgetJob(state.pendingJobId);
  state.job = null;
  state.pendingJobId = null;
  state.selectedBeat = null;
  state.selectedRow = null;

  $("results-table").querySelector("tbody").replaceChildren();
  $("results-body").hidden = true;
  $("results-empty").hidden = false;
  $("results-sub").textContent = "Upload a CSV to see the results here.";
  $("beat-caption").textContent = "Click a row (or use the arrow keys) to draw that beat.";
  $("threshold-note").textContent =
    "Moving the slider re-labels every result in your browser; nothing is uploaded again.";
  const download = $("download");
  download.href = "#";
  download.setAttribute("aria-disabled", "true");
  download.tabIndex = -1;
  $("clear-results").disabled = true;
  $("file-input").value = "";
  hideProgress();
  showError("");
  setBusy(false);
}

// ---------------------------------------------------------------- results

function loadJob(job) {
  state.job = {
    id: job.job_id,
    filename: job.filename,
    hasLabels: job.has_labels,
    threshold: job.summary.threshold, // the threshold the server labelled with
    rows: buildRows(job.results, job.has_labels),
    beatCache: new Map(),
  };
  $("results-table").classList.toggle("no-labels", !job.has_labels);
  $("confusion").hidden = !job.has_labels;
  $("stat-accuracy-tile").hidden = !job.has_labels;
  $("results-empty").hidden = true;
  $("results-body").hidden = false;
  const download = $("download");
  download.removeAttribute("aria-disabled");
  download.removeAttribute("tabindex");
  $("clear-results").disabled = false;
  updateThresholdUI();
  updateResults();
  if (job.results.length) selectBeat(1, false);
}

/** Build the table once; later threshold changes only touch the cells whose label flips. */
function buildRows(results, hasLabels) {
  const fragment = document.createDocumentFragment();
  const rows = results.map((r) => {
    const tr = document.createElement("tr");
    tr.dataset.beat = String(r.beat);
    tr.tabIndex = -1;
    const cells = [];
    for (let i = 0; i < (hasLabels ? 5 : 3); i += 1) cells.push(tr.insertCell());
    cells[0].textContent = String(r.beat);
    const prob = document.createElement("span");
    prob.className = "prob";
    const bar = document.createElement("span");
    bar.className = "prob-bar";
    bar.setAttribute("aria-hidden", "true");
    bar.appendChild(document.createElement("span")).style.width = `${r.probability * 100}%`;
    prob.append(r.probability.toFixed(4), bar);
    cells[1].appendChild(prob);
    if (hasLabels) cells[3].appendChild(pill(r.true_label));
    fragment.appendChild(tr);
    return { r, tr, predCell: cells[2], correctCell: hasLabels ? cells[4] : null, label: -1 };
  });
  $("results-table").querySelector("tbody").replaceChildren(fragment);
  return rows;
}

/** Re-label every beat at the current threshold and refresh all numbers. O(n), no requests. */
function updateResults() {
  const job = state.job;
  if (!job) return;
  const n = job.rows.length;
  let abnormal = 0, changed = 0, tp = 0, fp = 0, fn = 0, tn = 0;
  for (const row of job.rows) {
    const label = labelOf(row.r.probability);
    const isChanged = label !== row.r.label; // vs the server's default-threshold label
    abnormal += label;
    changed += isChanged ? 1 : 0;
    if (job.hasLabels) {
      const truth = row.r.true_label;
      if (label === 1) { if (truth === 1) tp += 1; else fp += 1; }
      else if (truth === 1) fn += 1; else tn += 1;
    }
    if (label === row.label) continue; // DOM already correct
    row.label = label;
    row.predCell.replaceChildren(pill(label));
    if (isChanged) {
      const tag = document.createElement("span");
      tag.className = "changed-tag";
      tag.textContent = "changed";
      tag.title = `Label differs from the default threshold ${job.threshold.toFixed(2)}`;
      row.predCell.appendChild(tag);
    }
    if (row.correctCell) row.correctCell.replaceChildren(verdict(label === row.r.true_label));
  }

  const normal = n - abnormal;
  const correct = tp + tn;
  $("stat-total").textContent = n.toLocaleString();
  $("stat-abnormal").textContent = abnormal.toLocaleString();
  $("stat-abnormal-pct").textContent = `${pct(abnormal, n)} of beats`;
  $("stat-normal").textContent = normal.toLocaleString();
  $("stat-normal-pct").textContent = `${pct(normal, n)} of beats`;
  if (job.hasLabels) {
    $("stat-accuracy").textContent = pct(correct, n);
    $("stat-accuracy-meta").textContent = `${correct} / ${n} correct`;
    $("cm-tp").textContent = String(tp);
    $("cm-fp").textContent = String(fp);
    $("cm-fn").textContent = String(fn);
    $("cm-tn").textContent = String(tn);
    $("cm-rates").textContent =
      `Sensitivity ${pct(tp, tp + fn)} · Specificity ${pct(tn, tn + fp)} · ` +
      `Precision ${pct(tp, tp + fp)}`;
  }
  $("stat-changed").textContent = changed.toLocaleString();
  $("stat-changed-meta").textContent = `vs default ${job.threshold.toFixed(2)}`;
  $("results-sub").textContent = `${job.filename} · threshold ${state.threshold.toFixed(2)}`;
  $("threshold-note").textContent = changed
    ? `${changed} of ${n} uploaded beats change label compared with the default ` +
      `threshold ${job.threshold.toFixed(2)} (marked "changed" in the table).`
    : `All ${n} uploaded beats keep the label they had at the default threshold ` +
      `${job.threshold.toFixed(2)}.`;
  drawSelectedBeat();
}

async function selectBeat(beat, focus) {
  const job = state.job;
  if (!job || beat < 1 || beat > job.rows.length) return;
  const tr = job.rows[beat - 1].tr;
  if (state.selectedRow) {
    state.selectedRow.classList.remove("selected");
    state.selectedRow.tabIndex = -1;
    state.selectedRow.removeAttribute("aria-selected");
  }
  state.selectedBeat = beat;
  state.selectedRow = tr;
  tr.classList.add("selected");
  tr.tabIndex = 0; // roving tabindex: Tab enters the table at the selected row
  tr.setAttribute("aria-selected", "true");
  if (focus) {
    tr.focus({ preventScroll: true });
    tr.scrollIntoView({ block: "nearest", behavior: reducedMotion.matches ? "auto" : "smooth" });
  }
  if (job.beatCache.has(beat)) { drawSelectedBeat(); return; }
  drawSelectedBeat(); // empty chart with "Loading…" until the values arrive
  const run = state.run;
  try {
    const data = await getJSON(`/jobs/${job.id}/beats/${beat}`);
    if (run !== state.run) return; // cleared meanwhile
    job.beatCache.set(beat, data.values);
    if (state.selectedBeat === beat) drawSelectedBeat();
  } catch (err) {
    if (run === state.run) $("beat-caption").textContent = err.message;
  }
}

function drawSelectedBeat() {
  const canvas = $("beat-canvas");
  const job = state.job;
  if (!job || !state.selectedBeat) {
    drawWave(canvas, null, 0, "", "Click a row to draw that beat");
    return;
  }
  const row = job.rows[state.selectedBeat - 1];
  const values = job.beatCache.get(state.selectedBeat);
  if (!values) {
    drawWave(canvas, null, 0, "", `Loading beat ${row.r.beat}…`);
    return;
  }
  const label = labelOf(row.r.probability);
  let title = `Beat ${row.r.beat}: ${TITLES[label]} · P(abnormal) ${row.r.probability.toFixed(4)}`;
  if (job.hasLabels) title += ` · true: ${row.r.true_label_name}`;
  drawWave(canvas, values, label, title);
  $("beat-caption").textContent =
    `Beat ${row.r.beat} of ${job.rows.length}. Line colour follows the prediction: ` +
    "blue ● normal, red ▲ abnormal. Click a row or use ↑/↓ to draw another beat.";
}

function setupTable() {
  const tbody = $("results-table").querySelector("tbody");
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (tr) selectBeat(Number(tr.dataset.beat), true);
  });
  tbody.addEventListener("keydown", (e) => {
    const job = state.job;
    if (!job || !state.selectedBeat) return;
    const moves = {
      ArrowDown: 1, ArrowUp: -1, PageDown: 10, PageUp: -10,
      Home: -Infinity, End: Infinity,
    };
    if (!(e.key in moves)) return;
    e.preventDefault();
    const target = Math.min(job.rows.length, Math.max(1, state.selectedBeat + moves[e.key]));
    selectBeat(target, true);
  });
}

// ---------------------------------------------------------------- start

document.addEventListener("DOMContentLoaded", async () => {
  setupTheme();
  setupDropzone();
  setupTable();
  $("threshold").addEventListener("input", onThresholdInput);
  $("threshold-reset").addEventListener("click", resetThreshold);
  $("sample-select").addEventListener("change", selectSample);
  $("sample-clear").addEventListener("click", clearSample);
  $("clear-results").addEventListener("click", clearResults);

  // Redraw the canvases whenever their box changes size (window resize, layout changes).
  let resizeFrame = 0;
  const observer = new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(redrawCharts);
  });
  for (const chart of document.querySelectorAll(".chart")) observer.observe(chart);

  updateThresholdUI();
  redrawCharts();
  await loadHealth();
  await loadSamples();
});
