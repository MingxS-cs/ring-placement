# Video-derived ring placement design

The user approved using the existing 119.4 s edited phone video, known dimensions, and rotation/noise augmentation to train a simulated robot. No additional recording or calibration is required for this version. The implementation remains inside this project directory.

## Data and geometry

Source: `C:/Users/minso/Downloads/359977334.mp4`, SHA-256 recorded in the sibling inspection directory. Candidate windows: `../inspection_359977334/episode_candidates.json`. The phone was fixed; the bearing was repositioned. Reset sections were removed and demonstrations concatenated. Never differentiate across cuts.

Bearing outer/inner diameter 50/40 mm, height 5 mm; ring outer/inner diameter 25/20 mm, height 7 mm. Both start horizontal on the tabletop. The success clearance for a horizontal ring is 7.5 mm radially. The final ring rests on the table inside the hole and protrudes 2 mm above the bearing.

Detect visible ring and bearing circles/ellipses in each candidate's clear initial and terminal frames. Validate overlays manually. Use bearing outer diameter as a local scale and record the result as an approximate, dimension-normalized mapping, not calibrated metric or 6D recovery. Video supplies initial positions, target positions, endpoint offsets and timing evidence. Height, fixed downward tool orientation and executable robot trajectories are planned in simulation and explicitly identified as synthetic.

## Simulation and policy

Use MuJoCo and a 7-DOF Panda model with narrow custom fingertips compatible with the ring's 2.5 mm wall. Prefer physical contact grasping and a free ring body; no object teleporting or forced attachment during evaluation. If a grasp simplification proves necessary, disclose it before reporting results.

Expose `RingEnv.reset(ring_xy, bearing_xy, seed)`, `observe()`, `step(action)`, `success()`, `render()` and `close()`. Robot low-level IK and joint position servos are shared by teacher/student. A transparent phase controller may select approach/grasp/lift/transfer/lower/release/retract; the learned model controls Cartesian corrections and gripper command. The student must not use teacher actions or teacher rescue during evaluation. Input features include robot/tool state, ring and bearing geometry, and phase. Actions and observations have explicit scales.

Generate robot demonstrations from video-derived configurations and consistent rotation/translation augmentation; add bounded observation noise and initial-state variation. Preserve dimensions. Store source episode ID and split with every simulated trajectory. Keep all descendants of an original episode in one split; reserve a bearing-position group as an additional challenge test if extraction supports three groups.

Train a small MLP with behavior cloning, train-only normalization and early stopping on validation. Use CPU with the installed CPU PyTorch; GPU is optional. If closed-loop drift needs correction, add teacher labels at student states only in the training split and report the intervention count.

## Acceptance and deliverables

Provide extraction metadata and overlays, deterministic split/augmentation manifests, a saved model checkpoint, raw per-episode closed-loop evaluation, a baseline using the same controller, and at least one rendered student rollout. Success requires release, full ring projection within the 40 mm hole, near-horizontal pose, table-level height and low speed held for a settling interval. Report the actual numerator/denominator, failure categories and strict criteria. Do not substitute BC loss for task success.

Provide runnable Python commands, dependencies, model provenance/licenses, README explaining real versus planned data and current limitations. A local Git project is prepared; external publication is a separate final action once the complete result is reviewable.
