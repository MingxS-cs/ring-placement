# Generalization Implementation Plan

Archived development plan. Paths and checklist states below describe the original planning stage; see the [historical index](../README.md) and [current method](../../method.md) for navigation.

**Goal:** Improve real-path prediction and validate simulated manipulation at new positions/directions/distances without reusing seen test outcomes for model selection.

**Architecture:** Add reviewed development motion, lock unseen transfer annotations, fit a low-capacity direction-conditioned ridge model, and execute its learned curve with physical feedback. Preserve V3 artifacts. Report learned fidelity separately from geometrically aided task success.

**Tech stack:** Existing Python/OpenCV/NumPy/MuJoCo stack, optional Matplotlib plots.

- [ ] Freeze independent annotation receipt for E20/E21/E22; keep labels hidden until selection freeze.
- [ ] Review additional genuine train/validation samples and validate their source/frame/path lineage.
- [ ] Add generalization_policy.py/generalization_train.py and meaningful regression tests for direction distinguishability, physical units, bounds, source isolation, fold-fitted statistics and rigid transforms with explicit reference frame.
- [ ] Add generalization_control.py with curve-arc retiming and tool-lag feedback; test bounded moves, progress tracking and exact endpoints.
- [ ] Freeze 96 new geometric scene configurations and objective/split/candidate grid before V4 training.
- [ ] Fit/select only on source-separated development folds, freeze model/source/protocol hashes, then open locked trajectory labels.
- [ ] Evaluate intermediate optical-path fidelity against straight and frozen V3; compare all three physical controllers on the same 96 configurations, retaining failures.
- [ ] Inspect annotation and robot render overlays, create deterministic demo/plots, and report actual improvements and limits.
- [ ] Run appropriate tests and clean-clone reproduction, perform independent method/code audit, rewrite main README with protocol history and bound generalization claims.
- [ ] Commit and produce a verified local V4 submission archive. Public publication remains deferred by the user.
