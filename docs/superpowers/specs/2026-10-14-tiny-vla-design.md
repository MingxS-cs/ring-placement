# Task-specific TinyVLA from the existing phone recording

User explicitly requests a small VLA implementation using the existing video, tested on new positions, directions and distances. Existing V4 and public-publication deferral remain unchanged. This authorized implementation uses a small network instead of a large pretrained VLA, enabling a concrete local baseline without multi-hour foundation-model fine-tuning.

## Model and data

Initial RGB256 tabletop image plus one of three text instructions generates 23 absolute Cartesian XYZ/gripper waypoints. The language commands mean place inside the bearing, 60 mm left, or 60 mm right in world/table X. A small full-resolution CNN predicts two object logit maps; learned spatial centroids and fixed camera calibration produce ring/bearing XY features. Word embeddings and a learned coefficient head condition the entire action sequence on actual text tokens. There is no runtime instruction-to-action if branch. Only a finite registered phrase vocabulary is supported; no open language understanding is claimed.

The ten scene features are ring XY, bearing XY, constant one, unit endpoint direction x/y and their three quadratic products. The action head is learned from retargeted synthetic action chunks. The video contributes the original source configurations and the train-only V4 fitted optical motion profile. Heights, grip commands and the left/right tasks are synthetic labels, not observations from the video. Synthetic pixel masks supervise vision during training only. Runtime images contain no mask or labels.

Use original 13 train and 3 validation source membership. Generate 16 training descendants per source and 8 validation descendants per source with rotation, bearing relocation and fixed dimensions. All descendants of each source remain in its split. Camera is fixed center XY=(.47,0), span .56 m, 256 squared orthographic pixels. A declared robot-hidden visual pass keeps table/object pixels visible; it is a simulator rendering convention, not a real camera or proof of handling robot occlusion. No object-centered crop is allowed.

Train vision using rendered pixel masks; train the text/waypoint head using the resulting source-derived action sequences. Model selection and all tuning use training/validation only. Retain validation placement outcomes and quantitative perception/waypoint errors. Record seed, hyperparameters and checkpoint hashes before the final test.

## Execution boundary

The generic executor follows neural waypoints through its own end-effector position, command position and finger opening, bounded Cartesian deltas and the unchanged IK/servos/contact physics. It advances when its own tool reaches the predicted waypoint, with declared minimum hold durations for grasp and release. This scheduling is hand-coded and must be disclosed: the result is a hierarchical task-specific VLA with a waypoint executor, not a pretrained general-purpose or fully autonomous low-level VLA.

The controller never receives ring/bearing true coordinates, env.phase, phase_target, teacher action, contact count or scoring results. Model inference occurs once at the start to create the complete action chunk; subsequent tracking is proprioceptive, with no visual replan after grasp. The V5 environment omits the original oracle-dependent FSM and keeps only physical integration and total step cap. True object state is used only for generation labels and outcome scoring.

Twenty-three waypoints: approach, descend, grasp, lift, sixteen transfer samples, lower, release, retract. Heights and minimum holding durations are documented execution/label conventions. Unknown instructions reject before rollout rather than silently mapping to a task.

## Frozen evaluation

Fresh seed 20261024: 96 procedural geometries, three bands 45–75/75–110/110–160 mm times eight angular bins times four placements. Use bearing bounds x=.40..52,y=-.10..10; condition geometric sampling so ring x=.32..62,y=-.23..23. No successful-outcome filtering, no invented human ancestors. These are new draws/combinations, not proof of unobserved direction sectors when training rotations already span the circle. The distance range includes short-distance extrapolation and interpolation; training video distances extend beyond160 mm.

Freeze scene/data/selection/source hashes and selected model before test execution. Primary full-image inside-task test96; image-black ablation96; matched-language counterfactuals on preset indices0,4,...92 with inside/left/right,72 attempts. Inside counterfactual repeats are explicitly already represented scenes, not independent trials. Keep all failures and exceptions in denominators. No teacher rescue, true-coordinate fallback or post-test model retuning.

Inside success uses original strict projected containment, horizontal/table-height, speed, open-finger/no-contact and0.4s settling criteria. Left/right success uses center error<3mm at bearing XY±(.06,0), with matching tilt/height/speed/release/settle criteria. Every task also requires the executor to complete all23 waypoints. Record actual ring/tool/action trajectories, model calls, predicted waypoints, input-image hash and final errors.

Image-black and same-scene language swaps test causal use of both modalities. Improved task success against V4 is not required; V4 was already at ceiling. Failures remain visible. All results are limited to the declared simulator camera/pass, object dimensions and task family; existing phone RGB to real robot deployment remains unverified.

## Reproduction and delivery

Preserve V4 files and archives. GPU training uses a separate local environment inheriting installed Torch2.11+CUDA12.8; record module paths/dependencies and this inheritance transparently. CPU inference remains portable where possible. Deliver runnable generation/training/evaluation/demo code, meaningful tests, all frozen outcomes, method/failure documentation and a local V5 archive. No GitHub publication.
