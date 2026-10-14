# Video motion version of the intern challenge

The user requested a complete version against the audited official challenge. Existing authorization covers the collected phone video, measured object dimensions, simulated manipulation, small-model training and augmentation. This version addresses the missing direct use of human motion and reproducibility. No additional recording is required.

## Requirements and alternatives

Official challenge: personally collected real manipulation data must drive a robotic manipulator in simple simulation. Hand-coded methods are allowed. Deliver a public GitHub repository with runnable instructions, example outputs, design decisions and successes/failures. Reference: https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221 (read 2026-10-14; deadline 2026-10-16 23:59 BST).

Three approaches were considered: (1) the previous scene-conditioned synthetic teacher, which uses real object locations but no human motion; (2) direct planar motion learning with physical grasp/lift/release planning; (3) full image-to-6D VLA retargeting, which cannot recover reliable depth/force from this uncalibrated monocular recording. Use approach 2. Preserve approach 1 as a labeled historical baseline. Do not claim approach 3.

## Data contract

Use only the original 359977334.mp4, hash 90cfbb066d2d88c1794ee7f7d517715f50a72dda8abf7248624650d0d4ca0dd4. Include it in the repository so extraction is self-contained. Use the existing 24 episode windows, never calculate differences across edited cuts. Track visible ring centers through movement with temporal consistency; observed, short-gap interpolation and missing values must be distinguishable. Exclude unreliable trajectories with recorded reasons. Save raw source frames, time, pixel centers, approximate bearing-relative XY and confidence, plus overlays for visual review.

Known-diameter normalization is approximate image-plane geometry, not camera calibration or 6D pose. Shadow and lift parallax remain limitations. Separate real observations from planned robot height/orientation/closure, endpoint anchoring and robot retiming.

## Learning and control

Train a tiny trajectory network directly on real motion. Express each measured XY path in a local start-to-end frame and learn its residual from a straight path. Enforce exact endpoint anchoring by a zero-ended envelope. Inputs are normalized progress and start-to-goal distance; outputs are local XY residuals. This transformation supplies translation/rotation equivariance without claiming extra independent demonstrations. Weight points by their arc-cell widths to reduce source-time sampling bias, then multiply observed/interpolated weights by 1/0.25 and normalize each episode to unit weight. This preserves real samples rather than relabeling synthetic resampling as observation. Select a checkpoint using validation sources only.

Runtime takes scene start/goal and robot progress, not a held-out human path. During transfer, the trained network generates the XY reference followed by bounded Cartesian actions and IK. Approach, descend, grip, vertical lift, lower, release and retract remain geometric, disclosed components. Count learned transfer steps and planned steps separately. The ring remains a free body, grasped by physical contact; no teleport or weld.

Do not modify the original environment/schema, so existing checkpoint bindings remain valid. New source bindings cover the motion file, policy/controller source and simulator. Test exact endpoint anchoring, rigid-transform equivariance, malformed/nonfinite data, split isolation and control bounds.

## Evaluation and delivery

Freeze 48 new simulated scene draws from the same 3 ordinary and 5 reserved source episodes before V3 training. These are correlated descendants, not new human recordings; prior V1/V2 outcomes informed the task design. Compare learned-motion hybrid with the same controller using zero path residual. Measure both strict physical task success and transfer-path change. Evaluate real held-out path prediction against straight interpolation after validation selection. Do not imply a success improvement if both methods solve the easy task.

Include failures in denominators, raw rollout outcomes, noisy scene/state stress if supported, deterministic first-case demo, source tracking overlays, source video, model, runnable source and licenses. Verify a clean clone with an isolated environment. Public GitHub publication is required for final delivery; inspect account and prepare reviewable local result before publishing. The job application itself and CV submission are outside this request.
