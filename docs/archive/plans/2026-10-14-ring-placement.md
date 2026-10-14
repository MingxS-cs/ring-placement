# Ring Placement Implementation Plan

Archived development plan. Paths and checklist states below describe the original planning stage; see the [historical index](../README.md) and [current method](../../method.md) for navigation.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute the tasks below in this session and review interfaces before integration.

**Goal:** Train and evaluate a small simulated robot controller using the existing ring/bearing video and geometry-consistent augmentation.

**Architecture:** Extract trusted visible endpoints into a provenance-rich episode JSON. Generate contact-based MuJoCo demonstrations from those configurations, split by original episode, train a compact policy, and evaluate independent closed-loop rollouts.

**Tech Stack:** Python 3.12, OpenCV, NumPy, MuJoCo, PyTorch, ffmpeg, pytest.

## Task 1: Video extraction

Files: `ring_learning/extract.py`, `tests/test_extract.py`, `data/real_episodes.json`, `outputs/extraction/`.

- [ ] Write and run a failing test that a known synthetic 50 mm bearing / 25 mm ring scene produces correct center separation and rejects missing object detections.
- [ ] Implement candidate-frame reading, bounded circle/ellipse detection, local dimension mapping and explicit missing/confidence labels. Read original PTS and use the existing candidate list.
- [ ] Run extraction on all candidates, save annotated initial/terminal frames, inspect overlays and correct or reject uncertain detections with recorded provenance.
- [ ] Verify no source window crosses an edited boundary and no output claims calibrated depth.

## Task 2: Robot simulation

Files: `ring_learning/sim.py`, `ring_learning/teacher.py`, `tests/test_sim.py`, `vendor/`.

- [ ] Write and run failing geometry/success tests: empty ring/bearing holes, outside placement rejected, ring at correct released pose accepted only after settling.
- [ ] Load an attributed Panda model; build annular collision geometry and narrow fingertips. Implement IK/servo control and physical object dynamics.
- [ ] Run one complete teacher grasp/lift/place/release rollout and inspect rendered images. Confirm lifted ring follows through contact and remains a free body after release.
- [ ] Provide a stable observation/action interface, record teacher state/action transitions and failure reasons.

## Task 3: Augmentation, training and evaluation

Files: `ring_learning/dataset.py`, `ring_learning/policy.py`, `ring_learning/train.py`, `ring_learning/evaluate.py`, `tests/test_dataset.py`, `tests/test_policy.py`.

- [ ] Write/run failing tests for split isolation, rotation of positions and Cartesian actions together, and train-only normalization.
- [ ] Split source episodes deterministically, augment only within each split and generate successful robot teacher trajectories. Save rejected teacher attempts as diagnostics.
- [ ] Train a compact MLP, save weights/scales/configuration, and compare with a geometry baseline under the same low-level controller.
- [ ] Evaluate unseen source episodes, an unseen bearing-position group and bounded perturbations without teacher action intervention. Save per-rollout success/failure data and a student video.

## Task 4: Reproduction and review

Files: `README.md`, `requirements.txt`, `outputs/results.json`, `outputs/student_rollout.mp4`.

- [ ] Verify commands on the actual runtime, all meaningful tests, source lineage, strict success metrics and rendered student behavior.
- [ ] Document limitations, planned heights/poses, augmentation bounds and baseline comparison. Record all executed versions and seeds.
- [ ] Run independent specification and code-quality reviews, address findings, then summarize the actual trained result to the user.

The source video is never modified. Test data or test outcomes are not used to tune the model; use validation for iterations and disclose any post-test exploration separately.
