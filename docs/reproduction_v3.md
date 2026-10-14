# Clean environment verification

## V3 full pipeline verification

On 14 October 2026, the isolated clone described below was updated to V3 Git revision `6d2f8fa3dbe98e38f4ee6f0c58296a1bd5909faf`. Its `.venv` still had `include-system-site-packages = false` and used the exact primary dependency versions in the README. The raw phone video is now a tracked repository file. No external source video or development-environment packages were needed for this V3 verification. FFmpeg/ffprobe remained host prerequisites.

The actual end-to-end command was:

```powershell
./scripts/reproduce_v3.ps1 -OutputDirectory outputs/v3_full_reproduction -PythonExecutable .venv/Scripts/python.exe
```

It completed with exit code 0 in **105.73 seconds**, using a new output directory and preserving delivered artifacts. It decoded the bundled video, regenerated endpoints and motion, froze scene configurations before training, trained a new model, ran all physical evaluation cases and rendered/encoded the demo. The included human visual-review decisions were reused; this check did not create new human annotations.

| V3 check | Actual result |
|---|---|
| Git byte preservation | Bundled video, endpoint JSON, motion JSON, delivered checkpoint and frozen scene JSON all equal their committed Git blobs byte for byte |
| `python -m pytest -q` | **137 passed, 0 skipped**, 6.47 seconds; the five original-video tests used the bundled file |
| Endpoint extraction | 24 accepted, 0 rejected; parsed endpoint JSON exactly equals the delivered endpoint data |
| Motion extraction | 9 accepted paths; all 24 regenerated motion episode records equal the delivered records |
| Training | 1,218 parameters; 5 train and 1 validation motion sources, 66 train and 10 validation points; selected epoch 9 |
| Model numerical repeatability | All 8 state-dictionary tensors, including input mean/std and network weights/biases, are **bit identical** to the delivered model |
| Scene repeatability | Complete train, validation, test and reserved-source-group configuration lists are identical |
| Learned motion physical evaluation | 18/18 test and 30/30 reserved-source-group scenes successful |
| Shared geometric straight-path baseline | 18/18 test and 30/30 reserved-source-group scenes successful |
| Full physical rollout repeatability | All 96 complete rollout records exactly equal the delivered evaluation records |
| Demo | Success, 209 steps; first frozen test scene; encoded video produced |
| `pip check` | No broken requirements found, both before and after optional plotting installation |
| Optional analysis | `matplotlib==3.11.2` installed from `requirements-plots.txt`; `video_analysis` completed on the included frozen model and produced both PNG figures and the analysis JSON |

The delivered checkpoint SHA-256 is `8a46863aa200b82726aee656e0732dc8ba0e00d7ea2f64f164af4f5eeedc0dea`; the freshly reproduced checkpoint is `1dd7a4708bb2a41a6651052a1d818519c2b67e2ab86a5a7b9d53515be8566178`. These different container hashes do not indicate different weights. The delivered endpoint JSON has `confidence_definition` in a different object-key position from the freshly generated JSON; parsed values are identical, but exact byte hashes differ. That difference propagates to motion's endpoint provenance and checkpoint metadata. The only changed checkpoint metadata fields are `real_episodes_sha256` and `motion_sha256`. Complete scene lists and every model tensor remain equal. The pipeline binds and evaluates the regenerated artifacts consistently rather than bypassing hash checks.

This reproduction replays the same frozen 48 synthetic scenes; it is **not a second independent benchmark**. Both controllers succeed 48/48, so this run establishes reproducibility and learned-path execution, not a learned advantage in placement success. Only planar transfer shape is learned; approach, grasp, vertical motion, release and other stages remain geometric. The source data remain a single edited phone recording, and no physical robot or public GitHub submission was validated.

Compact evidence is retained in `outputs/reproduction/v3_final`: full-pipeline console log and exit status, 137-test JUnit results, Git blob verification, exact tensor/configuration comparisons, evaluation summary, retrained model/history, demo outcome, optional analysis outcome and the complete installed dependency lock. Larger regenerated videos/images and full per-step rollouts remain in the isolated clone at `outputs/v3_full_reproduction`.

Additional commands executed:

```powershell
./.venv/Scripts/python.exe -m pytest -q --junitxml=repro_logs/pytest_v3.xml
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/python.exe -m pip install -r requirements-plots.txt
./.venv/Scripts/python.exe -m ring_learning.video_analysis --output outputs/v3_analysis_recheck
```

## Earlier V2 baseline verification

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
