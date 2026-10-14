# Ring placement from a phone recording

A simulated Panda arm picks up a 25 mm ring and places it inside a bearing with a 40 mm opening. The project uses a short, personally recorded phone video to build the training scenes and transfer curves.

The policy is a task-specific VLA trained from scratch: 33,930 parameters, one initial 256×256 RGB image and an English instruction in, 23 XYZ/gripper waypoints out. A programmed waypoint tracker and IK execute the plan. It supports nine phrases for inside, 60 mm left and 60 mm right; it does not understand unrestricted language or replan from live images.

## Data and assumptions

The [recording](data/source/359977334.mp4) lasts 119.4 seconds and contains 24 segmented demonstrations. The phone stayed fixed, the bearing had three placements, and reset footage was removed.

| Part | Where it comes from |
|---|---|
| Source layout and approximate planar motion | Phone video and reviewed annotations |
| Transfer curvature | V4 motion fit using only 13 training demonstrations |
| RGB, masks and robot-action labels | Simulation: 208 training scenes and 24 validation scenes, split by source demonstration |
| Height, grip timing, orientation and IK | Programmed conventions |

The visual encoder was trained on simulator RGB, not on the phone frames. Its fixed orthographic camera hides the robot geometry. Both objects start flat; dimensions, appearance and physics parameters stay fixed. Ring OD/ID/height: 25/20/7 mm. Bearing OD/ID/height: 50/40/5 mm.

## What was tested

V5 used 96 frozen combinations of position, eight transfer directions and 45–160 mm distance. Models were selected before these outcomes were opened.

| V5 experiment | Successful / registered attempts |
|---|---:|
| RGB + inside instruction | 96/96 |
| Same scenes and model, black RGB | 0/96 |
| Same 24 scenes, inside/left/right instructions | 72/72 |

The language experiment contains 24 repeats of the main task. Mean final center error was 0.507 mm, maximum 1.123 mm. These results show spatial generalization under the fixed simulator setup and dependence on both inputs. Training rotations already cover the circle; this is not a held-out angular-sector test or long-distance extrapolation.

The new [video-motion ablation](docs/video_motion_ablation.md) asks what the recorded transfer curve changes. It compares V5 with a straight-transfer readout, keeping RGB perception, text features, grasp/endpoints, Z, grip and execution identical. Both conditions still use video-derived endpoint geometry. A separate seed fixes 96 new scenes and 192 paired attempts before execution.

## Run

Use Python 3.12; ffmpeg/ffprobe are needed only for video rendering. The supplied model and the motion ablation run on CPU.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest -q
```

Repeat the original V5 benchmark or run the paired experiment into a new directory:

```powershell
./.venv/Scripts/python.exe -m ring_learning.vla_evaluate run --manifest outputs/vla_v5/configurations.json --receipt outputs/vla_v5/model_freeze.json --out outputs/vla_recheck.json
./scripts/run_motion_ablation.ps1 -OutputDirectory outputs/new_motion_ablation
```

For plots, install `requirements-plots.txt` and pass `-Plots` to the second command. Existing frozen outputs are preserved; choose a new destination. Training from scratch and the complete video-derived precursor pipeline are described in the [V5 baseline](docs/v5_baseline.md) and [V4 baseline](docs/v4_baseline.md).

## Evidence and limits

- [Six preset V5 replays](outputs/vla_v5/student_demo.mp4), [complete V5 rollouts](outputs/vla_v5/evaluation.json) and [freeze receipt](outputs/vla_v5/model_freeze.json).
- [Method](docs/tiny_vla_method.md), [Chinese overview](docs/tiny_vla_results_zh.md) and [separate-source reproduction](docs/reproduction_vla.md).
- Two failed text-head designs are retained under `outputs/vla_v5/model/attempt_01` and `attempt_02`. They are historical checkpoints, not compatible deployment models.

V4's learned curve did not beat straight interpolation on the reserved human paths. Placement success was already 96/96 for its straight baseline. The new ablation measures path effects without presuming a success advantage.

Phone-image perception, ordinary robot-visible camera input, new objects/materials and real hardware remain untested. The V5 reproduction regenerated identical datasets and weights, but Torch 2.11/2.14 physical trajectories differed slightly; the largest final ring-position difference was 0.0644 mm, with identical success decisions.

AI assisted annotation, code, experiments and writing; the applicant supplied the recording, dimensions and task choice. [Contribution details](docs/AI_ASSISTANCE.md).

Code: MIT. Panda assets: Apache 2.0, [attribution](vendor/NOTICE.md). This remains a local submission draft for the [Humanoid intern challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221); public GitHub publication is deferred.
