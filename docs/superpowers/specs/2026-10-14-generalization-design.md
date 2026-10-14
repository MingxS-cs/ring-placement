# Generalization to new position, direction and distance

The user requested actual generalization and confirmed the scope: new positions, directions and distances, with the same objects and fixed-camera recording. Continue using the existing phone video. Public publication remains deferred. Keep all V3 source/checkpoints intact.

## Evidence and chosen approach

V3 maps progress and endpoint length to a canonical path, discarding direction. Training episodes E02 and E04 have almost equal lengths but opposite local lateral residuals. It also fits normalized residual rather than physical displacement and has 1,218 parameters for five source paths. A training-only diagnostic supports a compact direction-conditioned, metre-displacement model. A broad candidate search was unstable; restrict production selection in advance.

Considered approaches: larger direction-aware MLP with augmentation, compact endpoint-anchored direction ridge, and shrinking endpoint kernel. Choose direction ridge because its six or twelve coefficients match the small source count and make extrapolation bounded and inspectable. Tune only polynomial degree 0/1 and ridge alpha 0.02/0.2/2. Candidate selection uses original-source leave-one-out folds of training sources, with each fold fitting its own direction statistics. Prefer six coefficients when validation differences are negligible. Fit error in physical approximate XY metres, not residual divided by distance.

## Data and blind evaluation

Add manually reviewed true ring centers for previously rejected TRAIN sources E03/E06/E07/E13/E15/E17/E18/E19 and VALIDATION sources E05/E11/E12 where visible. All label kinds and uncertainties must remain explicit; no guessed occluded positions or interpolation relabeled as observation. Prefer revised manual records to their older automatic records. The original source membership remains 13 train/3 validation/3 test/5 reserved.

Lock new transfer annotations for E20/E21/E22 as a blinded trajectory check. Training workers and root must not open label-bearing holdout files/images until the selected checkpoint, candidate selection protocol and physical scene manifest are frozen. A freeze receipt provides only source IDs/counts/hashes. These are newly annotated paths from already recorded reserved windows, with prior endpoint exposure and earlier V3 diagnostic use of other sources disclosed; they are not independent new recordings.

Use source-uniform arc-cell weights, observed/manual base weight 1 and old short-gap interpolated weight 0.25. Validate video hash, source identity, frame order/bounds and path normalization. Report intermediate-point displacement as well as whole-path displacement; exact endpoints must not inflate fidelity claims.

## Model and execution

Inputs are progress and start-to-goal direction in a defined camera/task reference frame. Predict local residual in metres with basis 4p(1-p) times [1,2p-1] (degree one) and [1,normalized direction x,normalized direction y]. Rotate local residual to the execution frame and cap its norm at 25 mm. Endpoints are exact. Global translation is structurally invariant. Rotational coordinate equivariance holds when the reference frame rotates with the scene; new directions under a fixed reference require empirical testing.

Retain the original contact-based RingEnv. Generate a reference curve from start/goal at transfer entry, parameterize execution by its own curve arc length, and advance at most 2 mm per control step. Pause reference advance when physical tool lag exceeds 4 mm. This avoids V3's chord-clock running ahead of a curved reference. Other stages, Z and gripper remain explicitly geometric. No ring attachment, state injection or teacher rescue is introduced.

## Test protocol and acceptance

Freeze 96 procedurally generated configurations: three distance bands 45–75,75–110,110–160 mm, eight angular bins covering the full circle, four placements per bin/band. Sample bearings within x=[0.40,0.52] m/y=[-0.10,0.10] m, condition reset geometry so rings remain within the predeclared conservative workspace x=[0.32,0.62] m/y=[-0.23,0.23] m. This is a declared workspace constraint, not a filter based on rollout outcomes. Keep every attempted case in denominators.

Compare V4, straight reference and the frozen V3 profile using identical new curve execution, stage controller, IK and physics. Report success by direction and distance, trajectory tracking, contacts and all failures. Novel procedural scenes have no invented human-source ancestry. No test outcomes select weights.

Desired evidence: lower unseen intermediate human-path error than straight/V3 on locked labels, stable placement across the frozen geometry range, and sensitivity checks under declared annotation uncertainty. Failure to achieve an advantage must remain visible. This establishes only same-task, same-object, fixed-reference simulated generalization and optical path fidelity; it cannot establish camera, lighting, object-size or real-robot generalization.

Measurement sensitivity is fixed before label access: seed 20261021, 100 draws per bound, independent uniform intermediate-center pixel perturbations within a square of half-width 5 px or 8 px. Preserve measured endpoints as anchors and recompute perturbed arc progress. Report this as a coordinate-uncertainty experiment, not a confidence interval, additional independent demonstrations or test-time retraining. Demonstration clips use preset case indices 0,16,32,48,64,80 and retain failures.
