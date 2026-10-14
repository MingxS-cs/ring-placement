# Video Motion Implementation Plan

> **For agentic workers:** Use subagent-driven-development for isolated motion extraction, neural trajectory learning and clean-environment verification, with root integration and independent final review.

**Goal:** Make personally collected video motion directly supervise a small policy that drives physical simulated ring placement, with a reproducible public submission.

**Architecture:** Preserve RingEnv and historical models. Add bounded episode tracking, a real-motion residual network and an explicitly hybrid controller. Freeze new evaluation cases before training and compare learned versus straight transfer on the same physics.

**Tech Stack:** Python 3.12, OpenCV, NumPy, PyTorch, MuJoCo, ffmpeg, Git/GitHub.

---

- [ ] Extract reviewed motion in ring_learning/motion.py, data/real_motion.json and outputs/motion, with tests/test_motion.py. Verify interpolation never crosses boundaries and distinguish observed evidence from inferred samples.
- [ ] Train direct real-motion network in ring_learning/video_policy.py and video_train.py, with tests/test_video_policy.py. Verify zero-ended paths and rigid-transform equivariance, reject invalid/source-leaking data, select only on validation.
- [ ] Freeze outputs/video_v3/configurations.json before model training. Preserve original source splits and record independent scene-draw seed and correlated-data limitation.
- [ ] Add ring_learning/video_evaluate.py and video_demo.py, count hybrid stages, compare learned and zero-residual transfer with identical control/physics, retain all failures and deterministic first-case output.
- [ ] Bundle raw video in data/source, make extraction/test source defaults repository-relative, and include trajectory overlays and source provenance.
- [ ] Verify all tests, fresh physical rollouts and clean-clone installation/evaluation. Inspect grasp/lift/release frames and compare extracted video tracks with actual moving ring.
- [ ] Rewrite README around V3 method and actual results, retain historical protocol documentation, include run commands, assumptions, limitations and official requirement mapping.
- [ ] Commit, build verified deliverable ZIP, publish public GitHub if authenticated access supports creation/push, verify public contents and clone. Report any unresolved external prerequisite explicitly.

Run verification commands from this repository with .venv/Scripts/python.exe; clean reproduction uses its own isolated venv without inherited packages. New regression tests must fail before the implementation that fixes them. Record actual outcomes rather than predicted success.
