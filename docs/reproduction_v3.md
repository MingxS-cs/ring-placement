# Clean environment verification

## Scope and revision

The initial verification below covers the committed V2 baseline at Git revision `fae2d6c65b8d8d6f66567df3a6dec8540d0bbfed`, before the V3 changes. It does not validate uncommitted or subsequently changed V3 code, datasets or checkpoints. A new V3 verification must identify its final revision and model hashes separately.

On 14 October 2026, a fresh local Git clone was created at `D:\model\ring_bearing_20261014\clean_reproduction_v3`. It contained 230 tracked files. Python 3.12.10 created a new `.venv` with `include-system-site-packages = false`; no packages from the development environment were inherited. The installation used the README's exact pinned requirements. Pip reused its download cache where available, so this checks isolation and dependency installation rather than uncached network download performance.

## Observed results

| Check | Actual result |
|---|---|
| CPU PyTorch install from the official CPU index | Exit 0, 64.62 seconds, `torch==2.14.0+cpu` |
| `pip install -r requirements.txt` | Exit 0, 40.34 seconds; no pin changes |
| Imports | NumPy, MuJoCo, Torch, OpenCV, Pillow and pytest all imported from the new virtual environment |
| `pip check` | No broken requirements found |
| `python -m pytest -q` | 103 passed in 6.20 seconds |
| First held-out test scene, trained student | 1/1 success, exit 0; model bindings verified |
| `python -m ring_learning.demo` | Exit 0, successful rendered and FFmpeg-encoded replay; 188 steps and 0 teacher actions |

Installed primary versions: NumPy 2.5.2, MuJoCo 3.12.0, Torch 2.14.0+cpu, opencv-python-headless 5.0.0.93, Pillow 12.3.0 and pytest 9.1.1. The complete transitive version list is in `outputs/reproduction/v2_baseline/environment-lock.txt`. FFmpeg and ffprobe were preinstalled on the host, version 8.1.1, and available on PATH. This was a Windows verification, not a Linux rendering verification.

The single-scene evaluation is a reproducibility smoke check, not an additional benchmark or a new success-rate claim. It replayed the first `test` configuration using the included checkpoint with SHA-256 `a01af2311f2f88a82c45946e38c01549dde26f96a57296a43bdfa4ff59d93773`.

## Commands executed from the clean clone

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/python.exe -m pytest -q --junitxml=repro_logs/pytest.xml
./.venv/Scripts/python.exe -m ring_learning.evaluate --output repro_logs/evaluation_smoke --split test --limit 1 --no-baselines --no-render
./.venv/Scripts/python.exe -m ring_learning.demo --output repro_logs/render_smoke/student_closeup.mp4
```

Machine-readable statuses, import locations, JUnit results, evaluation output, renderer outcome and compressed pip installation logs are retained in `outputs/reproduction/v2_baseline`. The isolated clone and environment were left intact for subsequent V3 verification.

## Baseline limitation

The V2 clone does not contain the raw phone video. Five tests in `tests/test_extract.py` read the original video at an absolute path on the user's machine; those tests passed here because that external file exists. Therefore, 103 passing tests do not establish that V2 extraction is self-contained. On a different machine these five tests skip when that original path is absent. Full extraction was not reproduced from the V2 clone alone. V3 should package or provide the actual source video, use repository-relative defaults, and verify extraction from that source.
