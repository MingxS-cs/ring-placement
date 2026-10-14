# Video-motion ablation implementation plan

**Goal:** Quantify the video-motion contribution and make the submission easier to read and explain, with honest assistance attribution.

**Architecture:** A train-only partial readout intervention plus a paired physical experiment. All existing V5 modules and sealed outputs remain byte-preserved.

**Tech stack:** Existing CPU PyTorch2.14, NumPy, MuJoCo, pytest and Matplotlib. No new model download or detector service.

## Train-only model intervention

Files: `ring_learning/motion_ablation_model.py`, `tests/test_motion_ablation_model.py`.

- [ ] Write tests before the module: only waypoint4..18 XY labels change; all other labels preserve bytes; shapes/finite inputs checked.
- [ ] Test that straight fitting changes only the final readout rows corresponding to waypoint4..18 XY; the input model is not mutated.
- [ ] Test shared visual/text features and critical actions, and linear transfer prediction within1e-6m on labeled geometry.
- [ ] Implement `linearize_transfer(chunks)` and `fit_straight_model(model, train_arrays)` using existing source-uniform coefficient fitting and phrase readout refinement. Restore all unchanged output rows directly from the original model.
- [ ] Record changed row indices, shared-state checks, coefficient rank and fitting error; no test/validation input to this function.

## Paired experiment

Files: `ring_learning/motion_ablation.py`, `tests/test_motion_ablation.py`.

- [ ] RED tests for96 deterministic fresh scenes/192 fixed requests, full strata, no overlap with V5 seed.
- [ ] RED tests for model/dataset/source/protocol tamper detection and refusing existing output paths.
- [ ] RED tests for exception/timeout denominators and paired RGB/critical-output checks; never filter to successful cases.
- [ ] Implement `prepare` to fit/freeze the straight checkpoint and save all bindings before any physical episode. The baseline uses only TRAIN arrays.
- [ ] Implement `run` using `VLAController` and `run_physical_case`; CPU threads=2. Both variants use exactly the same environment reset for each pair.
- [ ] Add actual airborne ring-path metrics and final center error. Keep complete trajectories and actual unsuccessful records.

## Evidence and prose

Files: `ring_learning/motion_ablation_analysis.py`, `README.md`, `docs/video_motion_ablation.md`, `docs/tiny_vla_results_zh.md`, `docs/AI_ASSISTANCE.md`, `docs/v5_baseline.md`.

- [ ] Preserve V5 README as a historical baseline with corrected relative links.
- [ ] Run model preparation, independent protocol review, freeze, and all192 attempts in that order; do not tune after opening outcomes.
- [ ] Produce paired path and outcome plots from saved records. Inspect rendered figures before including them.
- [ ] Rewrite English/Chinese explanations with measured results, data/control boundaries and actual failures. Do not fabricate applicant opinions or lower-detector-score claims.
- [ ] Run focused/full tests and validate the old V5 receipt; independently check paired results and preserved weights.
- [ ] Commit the local deliverable. Package to a new V6 ZIP and verify CRC and every member hash; publication remains deferred.
