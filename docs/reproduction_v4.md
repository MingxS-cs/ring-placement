# V4 clean-clone reproduction

## Revision and environment

The complete V4 pipeline was reproduced on 14 October 2026 from Git revision `1a9ac49bc16d05bdfa801ee64d6e70e3766c279c` in the separate clone `D:\model\ring_bearing_20261014\clean_reproduction_v3`. Tracked files were clean before checkout; earlier untracked V3 outputs were preserved. The clone's existing isolated Python 3.12.10 virtual environment was reused, with `include-system-site-packages = false`. This is a clean source-clone test on the same Windows host, not a claim of verification on another operating system or machine.

Primary runtime versions were NumPy 2.5.2, MuJoCo 3.12.0, Torch 2.14.0+cpu, opencv-python-headless 5.0.0.93, Pillow 12.3.0, pytest 9.1.1 and optional Matplotlib 3.11.2. Every third-party module imported from the isolated `.venv`; `ring_learning` imported from the separate clone. `pip check` reported no broken requirements. The recorded dependency lock includes transitive packages. FFmpeg and ffprobe 8.1.1 were preinstalled host dependencies on PATH.

The original phone video, endpoint and motion JSON, development and holdout annotation JSON, blind receipt, delivered policy, frozen scene manifest and model-freeze receipt were each compared against their Git blob bytes. All were preserved exactly.

## Commands and observed outcomes

```powershell
./.venv/Scripts/python.exe -m pytest -q --junitxml=repro_logs/pytest_v4.xml
./scripts/reproduce_v4.ps1 -OutputDirectory outputs/reproduced_v4_clean -PythonExecutable .venv/Scripts/python.exe
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/python.exe -m ring_learning.generalization_analysis --output outputs/v4_analysis_recheck
```

The complete script exited **0 in 259.25 seconds**. It was run without shortening or substituting pipeline stages. It extracted endpoints and optical motion from the bundled video, verified regenerated label values before rebinding byte-level provenance, froze 96 scenes, trained and froze the model, executed three physical controllers on all scenes, evaluated locked optical holdouts and measurement perturbations, then rendered six fixed demonstration clips and concatenated them.

Included visual-review decisions and manual development/holdout optical annotations were reused. Reproduction did not generate new human annotations or new real recordings. The regenerated data directory contains a copy of the original video with its original SHA-256.

| Verification | Actual result |
|---|---|
| Full tests | **193 passed, 0 failures, 0 errors, 0 skipped**, 8.26 seconds |
| Source measurements | 24 accepted endpoint episodes and 9 accepted automatic optical paths; endpoint JSON values exactly match delivered values |
| Training | 12 coefficients, degree 1, ridge alpha 0.02; 13 train sources/176 points and 3 validation sources/40 points |
| Selection repeatability | Complete training-source leave-one-out selection and candidate/fold results equal the delivered model |
| Model repeatability | Complete model values equal; coefficient, direction-mean and direction-std arrays have identical float64 bytes |
| Scene repeatability | All **96 complete configuration records** equal the delivered frozen manifest |
| V4 physical controller | 96/96 successful attempts |
| Straight physical baseline | 96/96 successful attempts |
| V3 profile physical baseline | 96/96 successful attempts |
| Physical rollout repeatability | All **288 complete rollout dictionaries**, including trajectories, actions and counters, equal delivered results |
| Optical holdout repeatability | 3 paths, 35 observed points, 29 intermediate points; complete report body excluding provenance equals delivered output |
| Sensitivity repeatability | All **200 draws** and resulting metrics equal delivered output: 100 paired draws at each of 5 px and 8 px radii |
| Demonstrations | All 6 preset cases succeeded; case metadata equals the delivered demo metadata |
| Video decoding | All 6 clips and the concatenated video decoded fully; 1,382 total frames, 20 fps, 960 × 414 pixels |
| Optional plots | Analysis command exited 0 and produced configuration coverage, heldout-path predictions and physical transfer comparison PNGs plus JSON |

The six demo indices were `0, 16, 32, 48, 64, 80`. Their decoded clip frame counts were `228, 201, 218, 223, 269, 243`, matching each case's executed steps plus 10 terminal hold frames. Demonstrations repeat existing benchmark cases and do not add trials to the denominator.

## Hash differences and binding checks

The delivered policy SHA-256 is `64310bf46ad153531358711719b74a12c4e1484f9f28b9bc1e4f204ba573f9a7`; the freshly trained policy SHA-256 is `b1009c78274c8c7a3edc8c6faee2a737079e3bd6177822a15561e2d6179986ee`.

Only the policy metadata fields `input_sources`, `motion_sha256` and `real_episodes_sha256` changed. Input paths point to the new output directory. Delivered and regenerated endpoint JSON parse to exactly the same values, but object-key order changes the byte hash; that hash change propagates to motion provenance. The reproduction helper checks original and regenerated label values before rebinding metadata, and never opens the blind trajectory annotations. Original blind labels are unchanged. Model values and source-uniform leave-one-out selection are identical.

Manifest changes are limited to endpoint/motion paths and their byte hashes. All scene records are identical. The fidelity report differs only in provenance; its complete numerical body, including every perturbation draw, is unchanged. Demo metadata differs only in checkpoint/manifest hashes. Receipt and model bindings were checked by the normal pipeline; no verification bypass was used.

## Interpretation

This run establishes reproducibility of the recorded-data-to-planar-motion pipeline, not an additional independent benchmark. All three controllers succeeded at every placement; the learned policy has no demonstrated placement-success advantage in this protocol.

The reproduced source-uniform mean error on the 29 intermediate optical points was **10.91 mm for V4**, **8.76 mm for the straight curve**, and **15.67 mm for V3**. These are approximate image-plane path errors using true observed endpoint anchors. V4 improves on the V3 profile in this holdout measure but does not beat the straight curve. The 5/8 px sensitivity draws are measurement perturbations, not confidence intervals or independent episodes.

Only XY transfer shape is learned. Grasp, vertical motion, fingers and other stages remain geometric. All human data come from the same edited fixed-camera recording; no calibrated 3D recovery or physical-robot transfer was verified. Work remained local, as requested; no public repository or application submission was attempted.

## Evidence

Compact evidence is in `outputs/reproduction/v4_final`: execution logs/status, 193-test JUnit output, runtime/module paths, dependency lock, Git blob verification, exact model/scene/rollout/fidelity comparisons, decoded-video checks, newly trained policy/history, regenerated freeze receipts and fidelity results. Full regenerated images, videos and the 57.6 MB per-step rollout JSON remain in the separate clone at `outputs/reproduced_v4_clean`; they were not duplicated into the evidence directory.
