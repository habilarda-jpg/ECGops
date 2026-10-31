# Profiling the training run (M4)

Machine: Windows 11, 4 logical CPUs (2 used by PyTorch), **3.4 GB RAM**, no GPU.
All numbers come from a 2-epoch run with MLflow enabled (local server from `docker-compose.yml`).

## Command

```bash
uv run python -m cProfile -o reports/profiling/train_before.prof -m ecgops.train training.epochs=2
# browse a profile interactively (snakeviz is in the dev group):
uv run snakeviz reports/profiling/train_before.prof
```

`.prof` files are git-ignored (`reports/profiling/`).

## Before: top 15 functions by cumulative time

```
   ncalls  tottime  cumtime  function
      225    0.039  465.660  torch/distributed/tensor/_ops/single_dim_strategy.py:627(_resolve_foreach...)
       53    0.002  418.031  torch/library.py:882(wrap)
       17    0.004  292.552  torch/_jit_internal.py:1047(_overload)
      902    0.006  147.422  torch/distributed/tensor/_ops/autogen.py:337(_find_foreach_variants)
  194/162    0.055  144.403  torch/library.py:272(define)
    34461   83.797   84.797  {built-in method nt.stat}
        2    0.002   84.069  mlflow/utils/requirements_utils.py:661(warn_dependency_requirement_mismatches)
      212    0.011   84.067  mlflow/utils/requirements_utils.py:609(_check_requirement_satisfied)
        1    0.006   83.335  mlflow/utils/environment.py:744(_process_pip_requirements)
      206    0.046   79.510  mlflow/utils/requirements_utils.py:473(_init_packages_to_modules_map)
      206    0.018   79.464  mlflow/utils/requirements_utils.py:445(_init_modules_to_packages_map)
        1    0.056   79.381  importlib_metadata/__init__.py:1159(packages_distributions)
        2    0.034   74.691  ecgops/train.py:80(train_one_epoch)
       70    0.046   69.059  importlib_metadata/__init__.py:1212(_top_level_inferred)
       70    0.002   68.956  importlib_metadata/__init__.py:588(files)
```

Total profiled time: 335 s (wall clock 380 s).

How to read it:

* The first five torch rows are **import-time registration code**. They exceed the total
  (465 s > 335 s) because cProfile double-counts recursive decorator calls. They are part of
  the import cost, not real work.
* The real hot spot is the MLflow chain `_process_pip_requirements` →
  `warn_dependency_requirement_mismatches` → `packages_distributions` (**84 s**). Every
  `mlflow.pytorch.log_model` call checks whether the model's requirements match the installed
  packages. To do that it reads the file list of every installed package and `stat`s ~9,000
  files. `nt.stat` alone is 84 s of self time because file access is slow on this machine
  (antivirus scanning, little free RAM). Timed in isolation the same scan took 58 s cold and
  5 s warm.
* Next, MLflow inferred the model's requirements by running `uv export` in a subprocess
  (**13 s**).
* The actual training (`train_one_epoch` ×2) is **75 s**.

**Largest contributor in our control:** our `mlflow.pytorch.log_model(...)` call. It took
~130 s from "Trained" to "Logged model" in the log, which is longer than the training itself.
Import time (`import mlflow` 38 s, torch/sklearn/pandas) is also large, but it comes from the
environment, not our code.

## What changed and why

1. **Explicit `pip_requirements`** (`model_pip_requirements()` in `train.py`). We pin `torch`,
   `numpy` and `cloudpickle` to the installed versions, dropping `+cpu` local labels so pip can
   resolve them from PyPI. MLflow then skips requirement inference and the `uv export`
   subprocess. Result: −13 s. The 84 s scan still ran, because MLflow runs it even for
   explicit requirements.
2. **Skip the redundant mismatch scan** (`skip_mlflow_requirement_check()`). The warning can
   never fire, because the requirements are pinned *from* the current environment. It is
   patched to a no-op only for the duration of `log_model`, in the two modules that call it.
   The patch is skipped with a log message if a future MLflow renames the function.
3. `log_model_seconds` is now logged as an MLflow metric, so the cost stays visible on every run.

Loadability check: a model registered with the new code (`models:/ecg-vgg1d/7`, logged from
inside Docker) was loaded back with `mlflow.pytorch.load_model`, and a prediction ran on a
test beat. That check exposed a separate bug, which is now fixed. MLflow 3 saves PyTorch
models with `torch.export`, which froze the batch size of the 1-beat input example: every
registered version so far accepts only batches of exactly one beat ("Guard failed:
x.size()[0] == 1"). The input example is now two real validation beats, which keeps the batch
dimension dynamic. Versions 1–7 have the batch-1 limitation and are tagged `superseded` in
the registry. **Version 8** (full 20-epoch run with the fix, `log_model` 46 s) loads back with
signature `(-1, 1, 187)`, predicts 5 beats in one call, and matches the local checkpoint
exactly.

## Before / after

| Run (2 epochs, MLflow on) | Before | + explicit requirements | + skip mismatch scan |
|---|---|---|---|
| `log_model` (log timestamps / metric) | ~130 s | ~125 s | **50.6 s** |
| requirement-mismatch scan | 84.1 s | 82.0 s | 0 s |
| `uv export` requirement inference | 13.0 s | 0 s | 0 s |
| `train_one_epoch` ×2 | 74.7 s | 73.8 s | 78.4 s |
| `import mlflow` | 37.9 s | 50.6 s | 35.8 s |
| total profiled time | 335 s | 399 s | 353 s |
| wall clock | 380 s | 435 s | not captured* |

\* The timer in the shell command failed for that run. Its `.prof` file was also overwritten
later by a run in which the MLflow server did not answer in time (that run trained without
MLflow: 215 s profiled, 256 s wall). A clean timed rerun was not possible because the PC ran
out of memory.

**Conclusion:** the fix removes ~80 s (about 60%) from model logging. Whole-run wall-clock time
is dominated by run-to-run noise in import times on this machine (±15 s for `import mlflow`
alone), so compare the `log_model` row, not the totals. Inside the Linux Docker image the
metadata scan is cheap anyway; there `log_model` took 104.6 s, mostly uploading through the
MLflow proxy on a 1.6 GB Docker VM.

## torch.profiler: one training epoch

```bash
uv run python -m ecgops.profiling
```

| Phase | Time | Share |
|---|---|---|
| dataloader | 3.8 s | 5.6% |
| forward | 34.0 s | 49.9% |
| backward | 27.5 s | 40.4% |
| optimizer (AdamW) | 2.8 s | 4.1% |

One epoch = 160 batches, 92 s with the profiler on (~37 s without it). The top operators are
`mkldnn_convolution` (35% of CPU time) and `convolution_backward` (29%). Batch norm,
max-pooling and ReLU together are under 12%. Data loading is not a bottleneck, so extra
DataLoader workers would not help. To train faster you would need fewer or narrower
convolutions (`model.channels`, `model.convs_per_block`) or a GPU.
