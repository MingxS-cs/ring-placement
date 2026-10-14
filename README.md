# Ring placement from a phone video

A small state-based policy controls a simulated 7-DOF Panda to pick up a ring and place it inside a bearing. The project uses 24 manually reviewed demonstrations from an edited, fixed-camera phone video. Known object dimensions provide an approximate planar scale; MuJoCo supplies executable robot trajectories and physical contact labels.

This is a hybrid controller: a transparent eight-stage state machine chooses the task stage; a learned MLP commands Cartesian displacement and gripper opening. IK, joint servos and gravity compensation are shared with the geometric teacher. The student receives no teacher action or rescue during evaluation.

## What comes from the real data

- Ring and bearing centers in visible initial and terminal frames, bearing image diameter, source frame indices/PTS and original episode IDs.
- Three distinct bearing placements. Start positions and bearing goals initialize the simulated scenes.
- Reviewed endpoint overlays and two recorded changes of terminal-frame selection where the original frame was partly occluded.

The mapping uses the bearing's 50 mm outer diameter and local image scale. It is not a calibrated reconstruction of millimetric 3D positions. Visible optical centers can be shifted by thickness, shadows and projection. Edited cuts are episode boundaries; velocities are never computed across them.

Robot trajectories, height, downward tool orientation, custom fingertip geometry and control actions are planned/generated in simulation. The teacher aims at the hole center; measured human endpoint offsets are retained as evidence, not forced into the robot target. Recorded human durations are metadata, not robot controller timing. This project does not train an image-to-action VLA or establish transfer to a real robot.

## Geometry and physical grasp

| Object | Outer diameter | Inner diameter | Height |
|---|---:|---:|---:|
| Bearing | 50 mm | 40 mm | 5 mm |
| Ring | 25 mm | 20 mm | 7 mm |

Both objects start horizontally on the tabletop. The ring is a free body throughout a rollout, with no weld or object teleporting. Narrow custom fingers pinch its 2.5 mm wall. Convex wedge segments preserve both holes in collision geometry. The ring rests on the table inside the hole and protrudes 2 mm above the bearing. Mass and friction are simulator assumptions, not measurements from the video.

Panda assets come from MuJoCo Menagerie at revision `d7b481bac63d3d600ab404ee74f7474c5efe8a67`, with Apache-2.0 attribution retained in [vendor/NOTICE.md](vendor/NOTICE.md).

## Reproduce

Tested with Python 3.12 on Windows, MuJoCo 3.12.0 and PyTorch 2.14.0+cpu. Install `ffmpeg` and `ffprobe` on PATH for video outputs. CPU training is sufficient for this model; CUDA is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Run commands from this repository directory. Reviewed endpoint data and the frozen configuration manifest are included, so training/evaluation do not require the source video.

```powershell
# Generate a new dataset from the included real endpoint measurements.
.\.venv\Scripts\python.exe -m ring_learning.generate --output outputs/reproduced_dataset --augment 16
.\.venv\Scripts\python.exe -m ring_learning.train --dataset outputs/reproduced_dataset --output outputs/reproduced_model --epochs 250
.\.venv\Scripts\python.exe -m ring_learning.evaluate --dataset outputs/reproduced_dataset --checkpoint outputs/reproduced_model/policy.pt --output outputs/reproduced_evaluation
```

To evaluate the delivered checkpoint directly:

```powershell
.\.venv\Scripts\python.exe -m ring_learning.evaluate --dataset outputs/dataset --checkpoint outputs/model/policy.pt --output outputs/recheck
```

To repeat extraction, provide the original video with the recorded SHA-256:

```powershell
.\.venv\Scripts\python.exe -m ring_learning.extract --source path/to/359977334.mp4 --candidates data/candidate_windows.json --visual-review outputs/extraction/visual_review.json
```

The raw source video remains outside this repository. Endpoint images contain the demonstrated objects and the operator's hand.

## Training and augmentation

Original sources are split before augmentation: 13 train, 3 validation, 3 ordinary test, and 5 from the third bearing-placement group as a challenge test. All descendants of a source remain in the same split. Train-only normalization and validation-only model selection are enforced.

Each training source has 16 configurations: the original plus bounded ±30° planar rotation, ±10 mm translation and ±1 mm start jitter. Rotate the ring's displacement about the bearing and translate both objects; preserve dimensions. Robot state/action labels are regenerated through physical teacher rollouts rather than rotating joint angles. Training also uses 0.2 mm standard-deviation position measurement noise, recomputing relative features consistently.

The model uses 21 features: ring minus tool position, bearing minus ring position, tool/ring velocities, gripper half-opening and eight stage indicators. Absolute coordinate columns and teacher-selected waypoint errors are excluded. Two 128-unit ReLU layers produce three bounded Cartesian displacements and one gripper command. Maximum displacement is 3 mm per 50 ms control step. The model is trained by weighted mean-square behavior cloning, with the checkpoint selected by validation loss.

## Evaluation

Evaluate the student in new physical rollouts. The positive control is the geometric teacher with the same IK, servo, phases and scene configurations. Zero-motion and untrained-MLP negative controls use one identity scene per held-out source; their different denominators are reported explicitly.

Success requires the complete ring projection inside the hole with a 0.3 mm margin, tilt at most 8°, table-height error at most 1.5 mm, linear speed below 0.01 m/s, angular speed below 0.15 rad/s, an open gripper, zero finger contacts and at least 0.4 seconds settling. A low cloning loss alone is not success.

Raw rollout outcomes and failure categories are saved in `outputs/evaluation/rollouts.json`; measured summaries in `outputs/evaluation/results.json`. The challenge group remains outside training bearing-position bounds. Its augmented rollouts are correlated descendants of five original sources, not independent new recordings.

## Files

- `data/real_episodes.json`: real endpoint measurements, confidence and provenance.
- `data/candidate_windows.json`: original episode-window references.
- `outputs/extraction/`: detection overlays, replayable visual review and boundary audit.
- `outputs/dataset/`: frozen split/augmentation configurations, teacher generation log and training arrays.
- `outputs/model/`: model checkpoint, normalization and training history.
- `outputs/evaluation/`: closed-loop results and student video.
- `ring_learning/`: extraction, simulation, teacher, generation, cloning and evaluation modules.

The model, deterministic seeds, environment versions and actual results are recorded in their associated JSON files. Learned control depends on the chosen simulator, fixed tool orientation and phase controller; it does not yet cover clutter, arbitrary tilted rings, unknown objects or real-world grasp transfer.
