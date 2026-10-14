# Ring placement from a phone video

A small state-based policy controls a simulated 7-DOF Panda to pick up a ring and place it inside a bearing. It uses 24 reviewed demonstrations from an edited fixed-camera phone video, known dimensions, geometry-consistent augmentation and physical MuJoCo demonstrations.

**Delivered model:** 21 → 128 → 128 → 4 MLP, **19,844 parameters**. It succeeded in **46/48 fresh simulated closed-loop scenes (95.8%)**, without teacher actions or rescue. See [student_closeup.mp4](../outputs/evaluation_final/student_closeup.mp4), [results.json](../outputs/evaluation_final/results.json) and [rollouts.json](../outputs/evaluation_final/rollouts.json).

| Controller | Fresh test descendants: 3 sources | Fresh challenge descendants: 5 sources | Total |
|---|---:|---:|---:|
| Learned student | 18/18 | 28/30 | 46/48 |
| Geometry teacher, same phases/IK/servo | 18/18 | 30/30 | 48/48 |
| Zero motion, open gripper | 0/3 | 0/5 | 0/8 |
| Untrained MLP | 0/3 | 0/5 | 0/8 |

Negative controls use one fixed scene per source. A separately labeled position-noise run also succeeded 46/48 times. Noise standard deviation is 0.2 mm, clipped to ±0.6 mm per raw measured coordinate; relative differences can vary by ±1.2 mm. Physical state is not modified by measurement noise.

The two primary failures were E21_ood_test_001 (approach timeout) and E23_ood_test_001 (descent timeout). These are retained, not removed from the denominator. No real-robot performance is established.

## Protocol history

V1 succeeded 18/18 ordinary trials but only 6/30 at the original third bearing point. Those diagnostics informed V2's wider augmentation, observable controller lag and train-only DAgger. V1 results remain in outputs/evaluation; its code is at Git revision daa6d4f and outputs/v1_source.zip.

Before V2 training, freeze 48 new seeded simulated scenes descended only from the same held-out real sources. Bearings are relocated within x=[0.40,0.52] m and y=[-0.10,0.10] m, preserving ring-to-bearing displacement. Final ood_test denotes the reserved source group; it is **not** testing at the original unseen bearing point or proof of spatial extrapolation. Expanded training positions overlap the final workspace.

These are correlated synthetic descendants, not 48 independent human recordings or an independent new real-data benchmark. The frozen manifest's phrase “Independent new scenes” refers only to new scene draws. [protocol_v2.json](../outputs/protocol_v2.json) records the clarification and relocation vectors. augmentation.translation_m describes the transform before relocation; final ring_xy_m/bearing_xy_m are the executed reset values.

Both V2 variants succeeded 24/24 validation rollouts; DAgger had lower validation cloning loss. Its hash was frozen in outputs/model_selection.json before final testing. Final outcomes were not used for further tuning.

## Real versus simulated data

The video supplies visible object centers, bearing image diameter, frame indices/PTS, endpoint offsets, original episode IDs and three bearing placements. All 48 endpoint overlays were reviewed. Two partly occluded terminal frames were replaced with recorded clear frames within their episodes.

The mapping uses the bearing's 50 mm outer diameter as a local planar scale. It is approximate dimension normalization, **not calibrated metric 3D recovery**. Thickness, shadow and projection can shift optical centers. Cuts are episode boundaries; no velocities are calculated across them.

Robot height, fixed downward orientation, custom fingers and executable state/action trajectories are planned/generated in simulation. The teacher aims at the hole center; human endpoint offsets/durations remain evidence rather than forced robot targets/timing. Inputs are simulated low-dimensional state, not raw images. Online camera-pose estimation and transfer to a physical arm are not validated.

## Geometry and physical grasp

| Object | Outer diameter | Inner diameter | Height |
|---|---:|---:|---:|
| Bearing | 50 mm | 40 mm | 5 mm |
| Ring | 25 mm | 20 mm | 7 mm |

Both start flat on the tabletop. Custom narrow fingers pinch the ring's 2.5 mm wall. The ring remains a free body throughout rollouts, with no weld, attachment or object teleporting. Convex wedge segments preserve both holes. The placed ring rests on the table and protrudes 2 mm above the bearing. Mass/friction are simulator assumptions.

Panda assets come from MuJoCo Menagerie revision d7b481bac63d3d600ab404ee74f7474c5efe8a67, with Apache-2.0 attribution retained in [vendor/NOTICE.md](../vendor/NOTICE.md). IK, joint servos and gravity compensation are shared between teacher and student.

## Reproduce

Tested with Python 3.12 on Windows, MuJoCo 3.12.0 and PyTorch 2.14.0+cpu. ffmpeg/ffprobe must be on PATH. CPU is sufficient; CUDA is optional. Run commands from the repository directory.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest -q
```

Evaluate the included final model without needing the source video:

```powershell
./.venv/Scripts/python.exe -m ring_learning.evaluate --output outputs/recheck
./.venv/Scripts/python.exe -m ring_learning.demo --output outputs/recheck/student_closeup.mp4
```

Regenerate the full pipeline in separate directories:

```powershell
./.venv/Scripts/python.exe -m ring_learning.generate --output outputs/reproduced_dataset --augment 32 --rotation 180 --translation .08 --configuration-seed 20261015 --fresh-tests
./.venv/Scripts/python.exe -m ring_learning.train --dataset outputs/reproduced_dataset --output outputs/reproduced_BC --epochs 250
./.venv/Scripts/python.exe -m ring_learning.dagger --dataset outputs/reproduced_dataset --checkpoint outputs/reproduced_BC/policy.pt --output outputs/reproduced_dagger --per-source 4 --beta .25 --seed 20261016
./.venv/Scripts/python.exe -m ring_learning.train --dataset outputs/reproduced_dagger --output outputs/reproduced_model --epochs 160 --initialize outputs/reproduced_BC/policy.pt
./.venv/Scripts/python.exe -m ring_learning.evaluate --dataset outputs/reproduced_dagger --checkpoint outputs/reproduced_model/policy.pt --output outputs/reproduced_evaluation
```

Repeat extraction with the original video matching the recorded SHA-256:

```powershell
./.venv/Scripts/python.exe -m ring_learning.extract --source path/to/359977334.mp4 --candidates data/candidate_windows.json --visual-review outputs/extraction/visual_review.json
```

The raw source video is outside the repository. Reviewed endpoint JSON/images are included. On Linux, rendering also needs a suitable MuJoCo OpenGL/EGL/OSMesa setup; the included Windows rollout was rendered and inspected.

## Learning

Original sources are split before augmentation: 13 train, 3 validation, 3 ordinary test and 5 reserved source-group cases. Every descendant stays on its source's side. Validation/test membership and NPZ row/configuration/source pairs are verified before optimization.

V2 plans 32 configurations per training source: original plus ±180° planar rotation, ±80 mm shared translation and ±1 mm ring-start jitter. Dimensions never scale. Actions are regenerated by physical teacher rollouts rather than rotating joint angles. Of 416 training attempts, 413 succeeded; three unreachable approaches were excluded and logged. All 24 validation teacher attempts succeeded, giving 437 successful demonstrations.

The 85,691 original training pairs are supplemented with 11,013 teacher labels at student-visited training states, totaling 96,704. DAgger visited 52 training scenes with a 25% teacher mixture, executing 2,665 teacher actions during collection. This is training-only intervention; evaluation intervention count is zero. The validation array and manifest were copied byte-for-byte, and the normalizer remained fitted on the original 85,691 training states.

The 21 features are ring minus tool position, bearing minus ring position, the student's integrated servo command minus measured tool position, ring velocity, gripper half-opening and eight phase indicators. Absolute coordinates and teacher waypoint errors are excluded. Two 128-unit ReLU layers produce three bounded Cartesian displacements and one gripper command, limited to 3 mm per 50 ms step. Training uses weighted MSE, bounded consistent measurement noise and validation checkpoint selection.

The state machine selects approach, descend, grasp, lift, transfer, lower, release and retract; it supplies no Cartesian motion. Lift completion checks actual vertical height/contact instead of returning to historical grasp XY. V2's teacher lifts vertically from its current command. The guard correction was diagnosed on validation before final testing.

Checkpoint bindings cover sim/teacher sources, observation/action schema and the frozen manifest. External model assets/runtime versions are separately attributed/recorded, rather than included in that source hash.

## Strict success

The complete ring projection must be inside the hole with 0.3 mm margin, tilt ≤8°, center height within 1.5 mm of flat table placement, linear speed <0.01 m/s, angular speed <0.15 rad/s, an open gripper, zero finger contacts and at least 0.4 s settling. Cloning loss or an open-gripper command alone does not prove success.

## Delivered files

- data/real_episodes.json and data/candidate_windows.json: source measurements and frame windows.
- outputs/extraction: reviewed overlays and boundary verification.
- outputs/dataset_v2 and outputs/dataset_dagger: frozen configurations, arrays and correction provenance.
- outputs/model_v2 and outputs/model_dagger: BC and selected final checkpoints/history.
- outputs/evaluation_final and outputs/noise_stress: strict outcomes, student videos and noisy-input results.
- outputs/model_selection.json and outputs/protocol_v2.json: model freeze and protocol clarification.
- ring_learning, tests and vendor: runnable code, verification and attributed assets.

The validated scope is horizontal single-object simulated placement with the chosen physics, custom fingers, fixed tool orientation and stage controller. Clutter, tilted starts, unknown objects and real-world grasp transfer remain untested.
