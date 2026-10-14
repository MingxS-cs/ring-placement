# Task-specific TinyVLA method

V5 is a 33,930-parameter vision/language/action baseline built for this task. It is trained from scratch, with no pretrained VLM or LLM. It supports nine registered English phrases covering three spatial targets: inside the bearing, 60 mm left, and 60 mm right in the fixed table X frame. It does not claim open vocabulary, arbitrary Chinese instructions or general-purpose robot understanding.

## What the phone recording contributes

The existing personally collected recording supplies object endpoints and reviewed approximate optical XY motion. The frozen train-only V4 direction-ridge model, fitted to13 original training-source paths, supplies transfer curvature. V5 rotates/relocates the original source endpoint geometry and uses that fitted motion to label a robot action chunk. There are208 training scenes (16 per source) and24 validation scenes (8 per each of3 validation sources). All descendants retain original source membership.

The phone's RGB pixels are not directly used to train V5's visual encoder. RGB training images are rendered from the reconstructed synthetic scenes. Object pixel masks, robot heights, grip values, left/right task labels and Cartesian actions are synthetic supervision. This is a transparent video-to-simulation distillation pipeline, not a dataset of personally recorded robot actions. The raw phone video and original/V4 evidence remain bundled.

## Vision, language and actions

The image is a fixed256×256 orthographic tabletop view, center(.47,0), span.56m. A small convolutional segmentation network predicts two foreground logit maps. Squared sigmoid probabilities produce spatial centroids; known synthetic-camera calibration converts them into predicted ring/bearing XY. The special pass hides robot geometry so it cannot occlude the objects; no object-centered cropping or runtime true masks are used. Ordinary demonstration views show the physical arm.

The10 geometric features contain predicted ring/bearing XY, constant1, unit relative direction and quadratic direction products. Train-only source-weighted least squares fit per-task target coefficient matrices for23 XYZ/gripper waypoints. Actual word-token embeddings feed a learned16→32 nonlinear text head and a linear coefficient output. Gradient training followed by a train-only last-layer least-squares refinement fits the nine phrase targets. Runtime language never selects an if/else action recipe or feeds an intent ID. This is finite supervised command grounding, not a pretrained semantic model.

The first design used a linear head on mean word embeddings. Validation showed that phrases of different lengths scaled left/right target differences inconsistently. A nonlinear hidden layer fixed the representation problem; final last-layer refinement removes residual coefficient accumulation. The two failed prototypes remain as archival weights/reports under `model/attempt_01` and `attempt_02`. Their old source bindings are intentionally incompatible with the current policy; they are not deployment checkpoints.

The visual network trains with pixel-mask BCE and calibrated-center losses,240 epochs on208 training images. Validation critical-waypoint XY error selects epoch146. No new96-scene outcome selects weights. Source-uniform weights prevent one original demonstration from dominating. Full matrix/centroid/perception metrics and training history are saved.

## Runtime boundary

One initial RGB image and the text command produce the entire23-waypoint plan: approach, descend, grasp, lift,16 transfer samples, lower, release and retract. A generic hand-coded executor tracks predicted XYZ with its own tool position and command state, caps displacement at3mm per50ms step and applies the predicted gripper value. Minimum hold durations and waypoint-reach conditions are execution conventions. No visual replan is performed after the initial image.

This is a hierarchical VLA with a waypoint executor, not a fully learned per-step control policy. The model emits all XYZ/gripper targets, but task decomposition, hold durations, IK, joint servos, camera calibration and execution safety bounds are hand-coded. Video does not recover depth, force, orientation, contact timing or gripper commands.

The V5 environment retains unchanged physical integration and removes the old target-dependent stage FSM. The runtime wrapper exposes RGB and seven own-proprioception values only. Object truth, contact count, environment phases, teacher actions and scoring are forbidden policy reads; attempted accesses raise an error and remain failures. Object state is initialized only during reset and otherwise follows contact physics. True state is used for training labels and scoring only.

## Validation evidence and precision limits

All24 validation scenes succeeded for all three primary commands (72/72), using only learned RGB/text plans. Mean/max grasp XY error were0.376/0.850mm; goal error0.637/0.922mm. With labeled training coordinates, text/critical-waypoint error is below0.001mm, while whole-curve approximation has0.908mm mean/6.956mm maximum XYZ error. That middle-path error is not hidden by the task success score.

An exact segmentation-mask centroid baseline already has roughly0.5mm mean optical raster error at this resolution;256 pixels across560mm gives2.1875mm per pixel. We therefore do not claim0.2mm localization on every scene. Camera/object appearance are fixed; generalization to phone images, new cameras, robot occlusion, lighting, sizes, friction or real hardware is unverified.

CPU Torch2.11 and2.14 yielded bit-identical predictions for all24 validation images. GPU/CPU XYZ differences were at most about0.016mm. This does not assert identical training across arbitrary hardware.

## Frozen test and causal controls

Fresh seed20261024 creates96 position/direction/distance combinations: three45–75/75–110/110–160mm bands, eight angular bins and four placements. Models, data, source, selected epoch and scene manifest are hash-bound before test execution. Training rotations cover the circle, so new angles are new draws, not withheld angular sectors. Training distances extend beyond160mm; this test is partly short-distance extrapolation and mostly novel combinations within support.

The full initial-image inside task has96 attempts. The same96 scenes with RGB replaced by zeros test image dependence. Preset indices0,4,...92 receive inside/left/right instructions (72 attempts), testing the same scene's language-dependent physical outcome.24 inside cases are repeats of the main benchmark, not independent observations. Exceptions/timeouts remain in denominators. Success requires completion of all23 waypoints plus strict settled placement; left/right must be within3mm of their corresponding target.

The main task's generous clearance means a geometric oracle can already perform well. The purpose is to demonstrate an actual RGB/text-conditioned action generator under the declared simulator conditions, not improvement over V4's ceiling or proof of general real-world VLA capability.

## Environment and sources

Training used RTX5080 with Torch2.11.0+cu128 in a separate `.venv-vla` inheriting installed Anaconda packages. V4's CPU environment was preserved. Extra inherited packages report dependency conflicts (such as unused numba); the actual Torch/NumPy/MuJoCo/OpenCV imports and CUDA tensor operations were verified. This is not a claim of a fully isolated clean GPU dependency environment. The portable inference model also runs in the original isolated CPU environment.

Reproduction uses the saved video-derived V4 precursor. To independently rebuild that precursor from raw video plus supplied annotations, run the historical V4 reproduction first. Reusing it does not create new human labels or independent data.

The foundation-model alternative has materially different scale: [official SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla) describes a450M model and a multi-hour training example. This implementation instead tests a much smaller task-specific multimodal baseline; it must not be presented as SmolVLA fine-tuning.
