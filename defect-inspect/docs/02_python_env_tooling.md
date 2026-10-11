# 02 — Python, Environments, and the Repo Layout

**Prerequisites:** none.

---

## 1. Concept

Python is the language the project is written in. But "writing Python" is only
half the story — the other half is controlling *which* Python and *which*
library versions run your code. That control is what a virtual environment
gives you, and it is the difference between a project that runs on the
examiner's machine and one that does not.

---

## 2. Why it is needed

### Why Python

| Reason | Consequence for this project |
|--------|------------------------------|
| Every deep learning framework targets Python first | PyTorch, ultralytics, ONNX Runtime all have Python as the primary API |
| The heavy work is not in Python | NumPy, OpenCV and ONNX Runtime are C++ underneath; Python only orchestrates |
| Raspberry Pi OS ships Python | No cross-compiler, no toolchain setup on the Pi |
| Readable | The examiner can follow the pipeline without knowing C++ |

The common objection is "Python is slow". It is, for loops over pixels. It is
irrelevant for us, because we never loop over pixels in Python — we call one
C++ function that processes the whole array. Module 03 shows the difference,
and it is a factor of about 100.

### Why a virtual environment

Install packages globally and you get one shared pile of libraries for every
project on the machine. Then:

- `unify-rul` needs `numpy 1.26`, this project needs `numpy 2.x`. One of them breaks.
- You upgrade ultralytics for another assignment. Your trained model no longer loads.
- You cannot tell the examiner what the project depends on, because the answer
  is "everything I have ever installed".

A virtual environment is a private folder holding one Python interpreter and
one set of packages, used by one project. `pyproject.toml` records exactly what
belongs in it, so the environment can be rebuilt anywhere from one command.

---

## 3. Real-world example

A factory commissions your inspection system in March. In September a
technician runs `pip install --upgrade` for an unrelated tool on the same Pi.
`numpy` jumps a major version, ONNX Runtime's binary interface no longer
matches, and the inspection line stops. With an isolated environment, the
unrelated tool gets its own packages and the line keeps running.

This is not hypothetical; it is why every production Python deployment is
containerised or virtualenv'd.

---

## 4. How it works

### The tool: uv

This project uses `uv`, a fast Python package manager. It replaces the older
`python -m venv` plus `pip` combination.

```
pyproject.toml          you declare what you need
      │
      │  uv sync
      ▼
uv.lock                 uv records the exact resolved versions
      │
      ▼
.venv/                  the private interpreter + packages
```

`uv sync` reads `pyproject.toml`, resolves versions, writes `uv.lock`, and
builds `.venv/`. Anyone who clones the repo runs one command and gets a
byte-identical environment.

### Runtime dependencies vs training dependencies

Look at our `pyproject.toml`. The split is deliberate:

```toml
dependencies = [        # installed everywhere, including the Pi
    "numpy", "opencv-python", "onnxruntime", "pyyaml", "flask", "pyserial",
]

[project.optional-dependencies]
train = [               # laptop only, never on the Pi
    "ultralytics", "torch", "torchvision", "onnx", "onnxslim", "matplotlib", "pandas",
]
pi = ["lgpio"]          # Pi only, will not build on Windows
```

`torch` is 2.6 GB. The Pi never trains anything, so it never needs torch. This
one split is what keeps the deployed system small and fast, and it is a
deliberate design decision worth stating in the viva.

---

## 5. Mathematical intuition

None. This module is engineering hygiene.

---

## 6. Software implementation

### Repo layout, and why each folder exists

```
defect-inspect/
├── pyproject.toml          what the project depends on
├── configs/default.yaml    every tunable number, in one file
├── data/                   datasets (git-ignored: too large, reproducible)
├── models/                 trained weights (git-ignored: large binaries)
├── src/inspector/          the importable library — the runtime system
│   ├── config.py           YAML -> validated dataclasses
│   ├── capture.py          get frames from a camera or a file
│   ├── preprocess.py       image -> model input tensor
│   ├── detector.py         run YOLOv8 via ONNX Runtime
│   ├── anomaly.py          PatchCore second opinion
│   ├── decision.py         detections -> PASS / FAIL
│   ├── storage.py          write results to SQLite
│   ├── actuator.py         fire the ejector (mock / serial / GPIO)
│   ├── pipeline.py         the loop that connects all of the above
│   └── bench.py            measure latency and frame rate
├── scripts/                things you run once, not part of the runtime
│   ├── fetch_dataset.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── build_patchcore.py
│   └── run_live.py
├── dashboard/app.py        the Flask web interface
├── firmware/ejector/       the Arduino sketch
├── tests/                  runnable checks
└── docs/                   this course
```

The rule separating `src/` from `scripts/`: **`src/` is imported, `scripts/` is
executed.** Anything in `src/inspector/` must run on the Pi. Anything in
`scripts/` may assume a laptop with a GPU.

### Why the package is called `inspector` and not `inspect`

`inspect` is a Python standard-library module. A top-level package named
`inspect` on the import path shadows it, and since PyTorch imports `inspect`
internally, the entire framework would fail with a confusing error. This is a
real bug that was caught before writing any code, and it is a good viva answer
about naming discipline.

### Commands you will actually use

```bash
uv sync --extra train        # build the environment, with training extras
uv run python scripts/train.py --epochs 100
uv run python -m inspector.bench --model models/model_int8.onnx
uv run python scripts/run_live.py --source data/samples/demo.mp4
```

`uv run` executes inside `.venv` without you having to activate anything, which
removes the single most common "it works on my machine" failure.

---

## 7. Hardware implementation

On the Raspberry Pi 5 the same repo is cloned, but the install is smaller:

```bash
uv sync --extra pi           # runtime + lgpio, no torch, no ultralytics
```

The Pi runs Debian on aarch64. Every runtime dependency listed above publishes
an `aarch64` wheel, which means installation is a download, not a compile. If
even one dependency lacked an ARM wheel, installation would fall back to
building from source, which on a Pi can take an hour. Checking wheel
availability *before* choosing a dependency is a deployment skill, not a
detail.

---

## 8. Common mistakes

- **Installing with system `pip`.** Pollutes the OS Python, and on Raspberry Pi
  OS it is now blocked by default (`externally-managed-environment`).
- **Committing `.venv/` or `data/` to git.** Both are large and both are
  reproducible from one command. Our `.gitignore` excludes them.
- **Using `conda`.** It resolves dependencies differently from pip and mixes
  the two silently. Pick one; this project uses uv only.
- **Hardcoding paths like `C:\Users\vishn\...` in the code.** The same file has
  to run on the Pi. All paths come from `configs/default.yaml` and are resolved
  relative to the repo root by `config.resolve()`.
- **Forgetting `--extra train` and then wondering why `import torch` fails.**

---

## 9. Viva questions

**Q: Why did you use a virtual environment?**
So the project's dependency versions are isolated, recorded and reproducible.
`uv sync` rebuilds the exact environment on any machine from `pyproject.toml`
and `uv.lock`, which is what makes the deployment to the Pi repeatable.

**Q: Python is slow. Why use it for a real-time system?**
Python is slow at interpreted loops, and we never write one in the hot path.
Every heavy operation — resizing, normalising, inference — is a single call
into compiled C++ inside OpenCV or ONNX Runtime. Python's cost is the
orchestration overhead, measured in microseconds against a ~40 ms frame budget.

**Q: What is the difference between `src/` and `scripts/`?**
`src/inspector/` is the deployable library, imported at runtime, and it is
constrained to dependencies available on the Pi. `scripts/` holds one-off
executables like training and export, which may assume a GPU laptop.

**Q: Why does the Pi not install PyTorch?**
Because it never trains. Inference runs through ONNX Runtime, which is roughly
50 MB against PyTorch's 2.6 GB, starts faster, and uses less RAM. Module 12
covers the export path.

**Q: How would another student reproduce your results?**
Clone the repo, `uv sync --extra train`, `python scripts/fetch_dataset.py`,
`python scripts/prepare_dataset.py`, `python scripts/train.py`. The dataset
split is seeded, so they get the same split we did.

---

## 10. Summary

1. Python is the orchestration layer; the compute is C++ underneath.
2. A virtual environment pins versions per project; `uv sync` rebuilds it anywhere.
3. Runtime dependencies and training dependencies are split on purpose — the Pi
   never installs torch.
4. `src/` is imported and must run on the Pi; `scripts/` is executed and may
   assume a GPU.
5. Never name a package after a standard-library module.

**Next:** [03 — NumPy and how a computer stores an image](03_numpy_images.md)
