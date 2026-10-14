# Controlled motion-ablation reproduction

## Initial source revision: formal fairness failure retained

The initial experiment at Git revision `fdd539b5fc4861aa158653fdcbd5186fd0c4da55` is **not an overall paired-experiment PASS**. Its formal 192-attempt run retained all records and both controllers succeeded 96/96, but only **95/96 pairs had identical initial RGB hashes**. The declared paired-input fairness criterion therefore failed. All common predicted outputs were equal across 96 pairs; that does not override the failed input-equality requirement. The first formal evaluation SHA-256 is `075e5038cc9ec3565cf2b39b39d6aea9592e0f5fa6b92ffc13f68dcae0a41b98`.

The verification below concerns baseline preparation and the first existing pair. A corrected observer/source revision and newly frozen scene draw are pending; no corrected full experiment is claimed here.

## Independent source clone and actual environment

An AI assistant performed automated reproduction from the separate clone `D:\model\ring_bearing_20261014\clean_reproduction_v3` on the same Windows host. Tracked files were clean before checkout, and previous untracked outputs were preserved. This was not a new human reproducer or applicant hand execution, and it adds no independent real recordings.

The clone used its isolated Python 3.12.10 `.venv` with `include-system-site-packages = false`: Torch 2.14.0+cpu, NumPy 2.5.2, MuJoCo 3.12.0, OpenCV 5.0.0, Pillow 12.3.0 and pytest 9.1.1. All third-party module paths point inside that virtual environment; `ring_learning` imports from the independent clone. CPU inference used two Torch threads, matching the formally declared experiment. `pip check` reported no broken requirements.

## Commands and verified scope

```powershell
./.venv/Scripts/python.exe -m pytest -q --junitxml=repro_logs/pytest_motion_ablation.xml
./.venv/Scripts/python.exe -m ring_learning.motion_ablation prepare --out outputs/motion_ablation_check
```

All **278 tests passed**, with zero failures/errors/skips, in 13.80 seconds. `prepare` exited 0 in 2.44 seconds. It rebuilt the straight-transfer baseline from the same frozen training arrays and sealed 96 scene records/192 paired requests before this clone's first-pair replay.

The reconstructed `straight_policy.pt` is **byte identical** to the formal baseline, SHA-256 `0fbfa01a9b3b4efbdb9d0229733bc625ba7e5a75b4aaba1ef711612910496dba`. All model tensors, buffers, checkpoint headers and metadata agree; maximum absolute tensor difference is zero. The complete `fit_report.json` is also byte identical, including fit diagnostics and source/changed-row evidence. All 96 scene records, 192 scheduled requests and shared-parameter audits match the formal preparation. Manifest differences are limited to model/binding output paths.

The clone then replayed only the first predetermined scene `M6_d0_a0_p0`, once per variant, through the public `run_case` and `VLAController` implementation after normal freeze/binding validation:

| Existing attempt | Outcome | Steps | Comparison to formal record |
|---|---|---:|---|
| `video_motion` | Success | 274 | Entire row exactly equal |
| `straight_transfer` | Success | 272 | Entire row exactly equal |

Both images had SHA-256 `22810cf7ff739c0098789c77bb70fecddefd6d1191232a2bea06809a53424ae9`. Complete neural chunks, all trajectory states/actions, transfer metrics, physical info and counters equal the corresponding formal rows. Their canonical row hashes are retained in the evidence. The full 192-attempt run was **not repeated in this clone**; two first-pair replays are repetitions of formal cases, not new independent trials or evidence that every pair passed.

## Interpretation and initial evidence

This is a controlled partial motion ablation: only intermediate transfer XY final-readout rows are refitted using TRAIN endpoint chords. Visual/text weights, critical actions, height/gripper behavior and source endpoint geometry remain shared. The straight variant still retains personally recorded geometry and inherited learned visual features; it is not a baseline without all real data.

Compact initial evidence is in `outputs/reproduction/motion_ablation_v6/initial_fdd539b`: tests/JUnit, runtime and import locations, preparation status, exact checkpoint/fit/scene comparisons, the two replay rows and their comparison to formal records, and the explicit failed formal-fairness summary. No production or sealed source/model/protocol was edited by this reproduction, and nothing was submitted or published.
