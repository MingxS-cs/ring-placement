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

The [video-motion ablation](docs/video_motion_ablation.md) compares V5 with straight transfer on 96 fresh paired scenes. Both succeeded 96/96. The actual ring path's mean maximum lateral departure was 11.91 mm with the video curve and 0.029 mm with straight transfer; mean path lengths were 100.59 and 94.32 mm. The video affects executed motion, but this easy placement task provides no success advantage for its curvature. Both variants still use video-derived endpoint geometry.

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
- [Paired paths](outputs/motion_ablation_v6_final/analysis/paired_ring_paths.png), [192 ablation rollouts](outputs/motion_ablation_v6_final/evaluation.json) and [ablation reproduction](docs/reproduction_motion_ablation.md). All 96 pairs share a saved initial RGB and identical non-transfer actions. The current suite has 281 passing tests.
- Two failed text-head designs are retained under `outputs/vla_v5/model/attempt_01` and `attempt_02`. They are historical checkpoints, not compatible deployment models.

V4's learned curve did not beat straight interpolation on the reserved human paths. The first V6 experiment also failed its strict image control in one pair because repeated rendering changed three background channel values by one gray level. Its records are retained. The corrected experiment shares one captured frame per pair and passes all 96 input/action checks, with unchanged model tensors.

Phone-image perception, ordinary robot-visible camera input, new objects/materials and real hardware remain untested. The V5 reproduction regenerated identical datasets and weights, but Torch 2.11/2.14 physical trajectories differed slightly; the largest final ring-position difference was 0.0644 mm, with identical success decisions.

AI assisted annotation, code, experiments and writing; the applicant supplied the recording, dimensions and task choice. [Contribution details](docs/AI_ASSISTANCE.md).

Code: MIT. Panda assets: Apache 2.0, [attribution](vendor/NOTICE.md). Public repository: [MingxS-cs/phone-demo-ring-placement](https://github.com/MingxS-cs/phone-demo-ring-placement). This project is prepared for the [Humanoid intern challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221); the internship application has not been submitted.
