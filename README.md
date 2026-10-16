# Ring placement

A simulated Panda arm picks up a 25 mm ring and places it inside a bearing with a 40 mm opening. A phone recording supplies the object layouts and approximate transfer paths.

The model has 33,930 parameters and is trained from scratch. It takes one initial 256×256 simulator image and an English instruction, then predicts 23 XYZ/gripper waypoints. A programmed tracker and IK execute those waypoints. It accepts nine fixed phrases for three targets: inside the bearing, 60 mm left, and 60 mm right. The model uses no further images during execution.

## Data

The [recording](data/source/359977334.mp4) contains 24 demonstrations over 119.4 seconds. The phone stayed fixed, the bearing had three placements, and reset footage was cut out. Ring outer/inner diameter and height are 25/20/7 mm; bearing dimensions are 50/40/5 mm.

| Training information | Source |
|---|---|
| Object layouts and approximate planar paths | Phone video and reviewed annotations |
| Transfer curve | Motion fit on 13 training demonstrations |
| RGB, masks and action labels | 208 training and 24 validation scenes rendered in simulation, split by source demonstration |
| Height, grip timing, orientation and IK | Programmed conventions |

The visual encoder trains on simulator images. The fixed orthographic camera hides the robot geometry. Both objects start flat; dimensions, appearance and physics parameters stay fixed.

## Results

The frozen benchmark covers 96 layouts spanning eight transfer directions and 45–160 mm distances. The model was selected before the tests were run.

| Test | Successes / attempts |
|---|---:|
| RGB + inside instruction | 96/96 |
| Same scenes and model, black RGB | 0/96 |
| Same 24 scenes, inside/left/right instructions | 72/72 |

The language test includes 24 repeats of the main task. Mean final center error on the main task was 0.507 mm, maximum 1.123 mm. Training already covers the full circle and longer distances, so these tests measure new layouts and direction/distance combinations, rather than held-out angles or long-distance extrapolation.

The [motion comparison](docs/motion_comparison.md) adds 96 new paired scenes. Video-derived curves and straight transfer both succeeded 96/96. Mean actual ring paths were 100.59 and 94.32 mm respectively: the video changes the executed path, but its curvature provides no success advantage here. Both conditions retain video-derived endpoint geometry.

![Actual ring paths in three preset scenes](outputs/motion_ablation_v6_final/analysis/paired_ring_paths.png)

See the [complete benchmark](outputs/vla_v5/evaluation.json), [paired rollouts](outputs/motion_ablation_v6_final/evaluation.json), [preset replays](outputs/vla_v5/student_demo.mp4) and [Chinese report](docs/results_zh.md). Phone-image control, a robot-visible camera, new objects/materials and real hardware remain untested.

## Run

Use Python 3.12. The supplied model and motion comparison run on CPU. Video rendering also needs ffmpeg/ffprobe.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest -q
```

Repeat the benchmark or motion comparison into a new destination:

```powershell
./.venv/Scripts/python.exe -m ring_learning.vla_evaluate run --manifest outputs/vla_v5/configurations.json --receipt outputs/vla_v5/model_freeze.json --out outputs/vla_recheck.json
./scripts/compare_motion.ps1 -OutputDirectory outputs/new_motion_comparison
```

For plots, install `requirements-plots.txt` and add `-Plots` to the second command. `scripts/reproduce.ps1` regenerates data, trains, evaluates and renders the preset demonstrations; its environment and commands are in [reproduction](docs/reproduce.md). Existing experiment outputs are preserved.

## Files

```text
ring_learning/   data processing, policies, control and simulation
scripts/         reproduction, motion comparison and model checks
data/            source recording and annotations
outputs/         saved datasets, models and experiments
docs/            method, results and reproduction reports
tests/           numerical, physics and evaluation checks
vendor/          Panda model and its license
```

Start with [vla_policy.py](ring_learning/vla_policy.py), [vla_control.py](ring_learning/vla_control.py) and the [method](docs/method.md). The [experiment index](outputs/README.md) identifies the current model and saved results. Earlier reports and scripts are in `docs/archive/` and `scripts/archive/`.

## Failed attempts and checks

Two failed text-head designs remain in `outputs/vla_v5/model/attempt_01` and `attempt_02`. The earlier curve fit also lost to straight interpolation on reserved human paths. The first paired experiment failed its image-equality check in one pair; sharing one captured frame per pair fixed this without changing model tensors. [The comparison report](docs/motion_comparison.md) retains both runs.

Separate-source reproduction regenerated identical training data and weights. Torch 2.11/2.14 contact trajectories differed slightly, with up to 0.0644 mm difference in final ring position and identical success decisions. The [policy](docs/reproduce.md) and [motion-comparison](docs/reproduce_motion.md) reproduction reports state exactly what was repeated.

The applicant supplied the recording, object dimensions and task choice. AI assisted annotation, code, experiments and writing; [contributions](docs/contributions.md) records that division.

Code: MIT. Panda assets: Apache 2.0, with [attribution](vendor/NOTICE.md). Prepared for the [Humanoid intern challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221); [submission checklist](docs/submission.md).
