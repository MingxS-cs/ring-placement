# Reproducing the motion comparison

The corrected experiment at `4536823b5961e4553969ffc78d41f91d1888d537`, seed 20261028, passed its formal 192-attempt paired-control checks: all 96 paired images and common predicted outputs agree, and both variants succeed 96/96. Independent same-host automated verification rebuilt the baseline exactly, passed 281 tests and replayed only the first pair with complete record equality. It did not repeat the full 192 attempts. The initial failed experiment and its evidence remain below.

## Initial source revision: formal fairness failure retained

The initial experiment at Git revision `fdd539b5fc4861aa158653fdcbd5186fd0c4da55` is **not an overall paired-experiment PASS**. Its formal 192-attempt run retained all records and both controllers succeeded 96/96, but only **95/96 pairs had identical initial RGB hashes**. The declared paired-input fairness criterion therefore failed. All common predicted outputs were equal across 96 pairs; that does not override the failed input-equality requirement. The first formal evaluation SHA-256 is `075e5038cc9ec3565cf2b39b39d6aea9592e0f5fa6b92ffc13f68dcae0a41b98`.

The verification below first records baseline preparation and the first existing pair from that initial source. The corrected observer/source revision and its newly frozen scene draw are documented in the corrected section below. The initial 95/96 failure remains retained.

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

## Corrected observer revision and scoped reproduction

The independent source clone was updated to Git revision `4536823b5961e4553969ffc78d41f91d1888d537`, preserving the initial output directory and evidence. The corrected protocol uses new predeclared scene seed **20261028**. Its runner captures exactly one initial RGB image per paired scene and passes that same captured observation to both variants. It archives the captured scene images. Policy weights are not retrained as part of this observer correction.

The same isolated Python 3.12.10 / Torch 2.14.0+cpu environment was used, with two Torch CPU threads. All **281 tests passed**, with zero failures/errors/skips, in 12.43 seconds. `pip check` again reported no broken requirements. A fresh `prepare --out outputs/motion_ablation_corrected_check` succeeded; the earlier directories were not reused or overwritten.

The freshly reconstructed straight baseline matches `outputs/motion_ablation_v6_final/straight_policy.pt` **byte for byte**, SHA-256 `419e41f47045962b7b83229ed13dd6bbb44f1fad68619fa7140f4302b71e6a38`. All tensors, buffers, checkpoint headers, metadata, fit-report bytes, 96 scene records, 192 scheduled requests and the shared-weight audit match the formal corrected preparation. The maximum tensor difference is zero. Manifest differences remain limited to model/binding output paths in the independent clone.

For the scoped physical replay, the clone explicitly reset the first predetermined scene once, called `render_policy()` once, and passed the same `initial_rgb` array to both public `run_case` calls. It did not let the variants independently capture their policy image. That image matches the first frame in the formal archived observation NPZ byte for byte.

| Corrected first existing attempt | Outcome | Steps | Formal-row comparison |
|---|---|---:|---|
| `M6_d0_a0_p0:video_motion` | Success | 269 | Entire row exactly equal |
| `M6_d0_a0_p0:straight_transfer` | Success | 253 | Entire row exactly equal |

Both policy-image hashes are `c15b770c3b87bee9f387386d71de00b0684d393e8f8a8d5de13b060bc280a48c`. Neural chunks, complete trajectories, transfer metrics, physical info and counters match the corresponding formal records exactly. Canonical row SHA-256 values and the single captured frame are retained in the corrected evidence.

The formal corrected evaluation file contains all **192 attempts** and reports `paired_control_checks_passed = true`: **96/96 identical initial images**, **96/96 identical common predicted outputs**, and **96/96 successes for each variant**. Its SHA-256 is `248964853e6d3d9ea60bf0f2f8733b34b7718aeade891dbc4fe57f2fb4365541`. The independent clone read this completed report and compared its first pair; it did **not** rerun all 192 attempts. Equal placement success is not evidence that the video-shaped path improves task completion over the straight path.

Corrected compact evidence is in `outputs/reproduction/motion_ablation_v6/corrected_4536823`: new tests/JUnit, preparation/runtime records, exact baseline and scene comparisons, the two complete replay rows, one captured RGB frame and the formal-row comparison. This remains same-host **AI automated reproduction**, not applicant manual execution, another human reproducer, a new independent benchmark or additional real data. All initial-failure evidence and the earlier 278-test account remain intact; frozen production source, specifications, models and formal outputs were not edited by this reproduction.
