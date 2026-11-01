// ECGops web page: talks to the FastAPI service in src/ecgops/api.py.
// Plain JavaScript, no build step and no external libraries, so it works offline.
"use strict";

const SAMPLING_RATE_HZ = 125;
const POLL_INTERVAL_MS = 300;

const state = {
  threshold: 0.5,        // current slider value
  serverThreshold: 0.5,  // default threshold reported by /health
  samples: [],
  sample: null,          // { beat, probability } for the selected sample
  job: null,             // { id, results, hasLabels, threshold }
  selectedBeat: null,
};

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const labelOf = (p) => (p >= state.threshold ? 1 : 0);
const NAMES = ["normal", "abnormal"];

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

function pill(label) {
  const span = document.createElement("span");
  span.className = `pill ${NAMES[label]}`;
  span.textContent = NAMES[label];
  return span;
}

/** Draw one beat on a canvas: time axis in seconds, amplitude 0..1, red/blue by label. */
function drawWave(canvas, values, label, title) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 800;
  const height = Math.round(width * 0.3) || 240;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 36, right: 10, top: title ? 24 : 10, bottom: 24 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const seconds = (values.length - 1) / SAMPLING_RATE_HZ;
  const x = (i) => pad.left + (i / (values.length - 1)) * plotW;
  const y = (v) => pad.top + (1 - v) * plotH;

  // Grid and axis labels.
  ctx.strokeStyle = css("--grid");
  ctx.fillStyle = css("--muted");
  ctx.font = "11px system-ui, sans-serif";
  ctx.lineWidth = 1;
  for (let v = 0; v <= 1.0001; v += 0.25) {
    ctx.beginPath(); ctx.moveTo(pad.left, y(v)); ctx.lineTo(width - pad.right, y(v)); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(v.toFixed(2), pad.left - 4, y(v) + 4);
  }
  ctx.textAlign = "center";
  for (let t = 0; t <= seconds + 1e-9; t += 0.25) {
    const px = pad.left + (t / seconds) * plotW;
    ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, pad.top + plotH); ctx.stroke();
    ctx.fillText(`${t.toFixed(2)} s`, px, height - 6);
  }

  // Waveform.
  ctx.strokeStyle = css(label === 1 ? "--abnormal" : "--normal");
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  values.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
  ctx.stroke();

  if (title) {
    ctx.fillStyle = css("--text");
    ctx.textAlign = "left";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText(title, pad.left, 16);
  }
}

// ---------------------------------------------------------------- health + threshold

async function loadHealth() {
  const status = $("model-status");
  try {
    const health = await getJSON("/health");
    state.serverThreshold = state.threshold = health.threshold;
    $("threshold").value = health.threshold;
    $("threshold-value").textContent = health.threshold.toFixed(2);
    if (health.model_loaded) {
      status.textContent = `Model ready: ${health.model}`;
    } else {
      status.textContent = `Model not loaded: ${health.error}`;
      status.classList.add("bad");
    }
  } catch (err) {
    status.textContent = `API not reachable: ${err.message}`;
    status.classList.add("bad");
  }
}

function onThresholdChange() {
  state.threshold = Number($("threshold").value);
  $("threshold-value").textContent = state.threshold.toFixed(2);
  renderSample();
  renderResults();
}

// ---------------------------------------------------------------- samples

async function loadSamples() {
  const select = $("sample-select");
  try {
    const data = await getJSON("/samples");
    state.samples = data.beats;
    select.innerHTML = "";
    for (const beat of data.beats) {
      const option = document.createElement("option");
      option.value = String(beat.id);
      option.textContent = `Beat ${beat.id}` + (beat.label_name ? ` (true: ${beat.label_name})` : "");
      select.appendChild(option);
    }
    select.disabled = false;
    await selectSample();
  } catch (err) {
    select.innerHTML = "<option>No samples available</option>";
    $("sample-result").textContent = err.message;
  }
}

async function selectSample() {
  const beat = state.samples.find((b) => String(b.id) === $("sample-select").value);
  if (!beat) return;
  state.sample = { beat, probability: null };
  renderSample();
  try {
    const result = await getJSON("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ beats: [beat.values], threshold: state.threshold }),
    });
    if (state.sample && state.sample.beat === beat) {
      state.sample.probability = result.predictions[0].probability;
      renderSample();
    }
  } catch (err) {
    $("sample-result").textContent = err.message;
  }
}

function renderSample() {
  if (!state.sample) return;
  const { beat, probability } = state.sample;
  const out = $("sample-result");
  if (probability === null) {
    drawWave($("sample-canvas"), beat.values, beat.label ?? 0, `Beat ${beat.id}`);
    out.textContent = "Predicting…";
    return;
  }
  const label = labelOf(probability);
  drawWave($("sample-canvas"), beat.values, label,
    `Beat ${beat.id}: ${NAMES[label]} (P(abnormal) = ${probability.toFixed(4)})`);
  out.replaceChildren(
    "Prediction: ", pill(label),
    `  P(abnormal) = ${probability.toFixed(4)}`,
  );
  if (beat.label !== null && beat.label !== undefined) {
    const verdict = beat.label === label ? "correct" : "wrong";
    out.append(`  ·  true label: ${beat.label_name} (${verdict})`);
  }
}

// ---------------------------------------------------------------- upload + progress

function showProgress(text, percent, done = false) {
  $("progress-area").hidden = false;
  $("progress-area").classList.toggle("done", done);
  $("progress-text").textContent = text;
  $("progress-fill").style.width = `${percent}%`;
  $("progress-bar").setAttribute("aria-valuenow", String(percent));
}

function showError(message) {
  const el = $("upload-error");
  el.textContent = message;
  el.hidden = !message;
}

/** Phase 1: upload with real XMLHttpRequest progress events. Resolves with {job_id, total}. */
function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/jobs");
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((100 * event.loaded) / event.total);
        showProgress(`Uploading… ${percent}%`, percent);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new Error(errorMessage(xhr.response, xhr.status)));
    };
    xhr.onerror = () => reject(new Error("Network error: is `uv run serve` still running?"));
    xhr.send(form);
  });
}

/** Phase 2: poll GET /jobs/{id} until the background job is done (or failed). */
async function pollJob(jobId) {
  for (;;) {
    const job = await getJSON(`/jobs/${jobId}`);
    if (job.status === "error") throw new Error(`Analysis failed: ${job.error}`);
    showProgress(`Analysing… ${job.processed}/${job.total} beats (${job.percent}%)`, job.percent,
      job.status === "done");
    if (job.status === "done") return job;
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
}

async function handleFile(file) {
  if (!file) return;
  showError("");
  $("results").hidden = true;
  showProgress("Uploading… 0%", 0);
  try {
    const created = await uploadFile(file);
    showProgress(`Analysing… 0/${created.total} beats (0%)`, 0);
    const job = await pollJob(created.job_id);
    state.job = {
      id: job.job_id,
      filename: job.filename,
      results: job.results,
      hasLabels: job.has_labels,
      threshold: job.summary.threshold,
    };
    state.selectedBeat = null;
    renderResults();
    $("results").hidden = false;
    if (job.results.length) selectBeat(1);
  } catch (err) {
    $("progress-area").hidden = true;
    showError(err.message);
  }
}

function setupDropzone() {
  const zone = $("dropzone");
  const input = $("file-input");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => { handleFile(input.files[0]); input.value = ""; });
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  }
  for (const type of ["dragleave", "drop"]) {
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove("dragover"); });
  }
  zone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
}

// ---------------------------------------------------------------- results

function renderResults() {
  const job = state.job;
  if (!job) return;
  const results = job.results;
  let abnormal = 0, correct = 0, changed = 0;
  const rows = document.createDocumentFragment();
  for (const r of results) {
    const label = labelOf(r.probability);
    abnormal += label;
    if (label !== r.label) changed += 1;
    const tr = document.createElement("tr");
    tr.dataset.beat = String(r.beat);
    if (label !== r.label) tr.classList.add("changed");
    if (state.selectedBeat === r.beat) tr.classList.add("selected");
    const cells = [String(r.beat), r.probability.toFixed(4), NAMES[label]];
    if (job.hasLabels) {
      const ok = r.true_label === label;
      correct += ok ? 1 : 0;
      cells.push(r.true_label_name, ok ? "yes" : "no");
    }
    cells.forEach((text, i) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (i === 2) td.className = NAMES[label];
      if (i === 4) td.className = text === "yes" ? "right" : "wrong";
      tr.appendChild(td);
    });
    rows.appendChild(tr);
  }
  $("results-table").querySelector("tbody").replaceChildren(rows);
  $("results-table").classList.toggle("no-labels", !job.hasLabels);

  const n = results.length;
  let summary = `${abnormal} of ${n} beats abnormal (${((100 * abnormal) / n).toFixed(1)}%) ` +
    `at threshold ${state.threshold.toFixed(2)}.`;
  if (job.hasLabels) summary += ` Accuracy: ${((100 * correct) / n).toFixed(1)}% (${correct}/${n}).`;
  $("summary").textContent = `${job.filename}: ${summary}`;

  $("threshold-note").textContent = changed
    ? `${changed} of ${n} uploaded beats change label compared with the server threshold ` +
      `${job.threshold.toFixed(2)} (shown in italics).`
    : "Moving the slider re-labels the results in your browser.";
  $("download").href = `/jobs/${job.id}/results.csv?threshold=${state.threshold}`;
  if (state.selectedBeat) drawSelectedBeat();
}

async function selectBeat(beatNumber) {
  state.selectedBeat = beatNumber;
  for (const tr of $("results-table").querySelectorAll("tbody tr")) {
    tr.classList.toggle("selected", tr.dataset.beat === String(beatNumber));
  }
  try {
    const data = await getJSON(`/jobs/${state.job.id}/beats/${beatNumber}`);
    state.job.beatValues = { beat: beatNumber, values: data.values };
    drawSelectedBeat();
  } catch (err) {
    $("beat-caption").textContent = err.message;
  }
}

function drawSelectedBeat() {
  const cached = state.job && state.job.beatValues;
  if (!cached || cached.beat !== state.selectedBeat) return;
  const r = state.job.results[cached.beat - 1];
  const label = labelOf(r.probability);
  let title = `Beat ${r.beat}: ${NAMES[label]} (P(abnormal) = ${r.probability.toFixed(4)})`;
  if (state.job.hasLabels) title += `, true: ${r.true_label_name}`;
  drawWave($("beat-canvas"), cached.values, label, title);
  $("beat-caption").textContent = "Click a row to draw that beat. Red = abnormal, blue = normal.";
}

// ---------------------------------------------------------------- start

document.addEventListener("DOMContentLoaded", async () => {
  $("threshold").addEventListener("input", onThresholdChange);
  $("sample-select").addEventListener("change", selectSample);
  $("results-table").querySelector("tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (tr) selectBeat(Number(tr.dataset.beat));
  });
  window.addEventListener("resize", () => { renderSample(); drawSelectedBeat(); });
  setupDropzone();
  await loadHealth();
  await loadSamples();
});
