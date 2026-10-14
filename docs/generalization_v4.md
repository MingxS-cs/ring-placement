# V4 generalization design and interpretation

The requested scope is new object positions, start-to-goal directions and distances, using the same ring, bearing and fixed-camera recording. The implementation tests a hybrid motion controller in contact-based MuJoCo physics. It does not test a visual detector on novel rendered scenes or a physical robot.

## Why change V3

V3 gave its 1,218-parameter network only progress and endpoint distance. E02 and E04 differ by only 0.437 mm in distance but approximately 90 degrees in direction; their measured local lateral residuals have opposite signs. The original inputs cannot distinguish that pair. Scaling a normalized residual by a novel distance also expands uncertain optical curvature. V3's saved negative results remain in [the historical baseline](v3_baseline.md).

V4 retains direction information and predicts residuals in approximate metres directly. A six-candidate ridge family limits capacity: degree zero or one, with ridge penalty 0.02, 0.2 or 2. Candidate selection uses leave-one-original-training-path-out error. Every fold fits direction normalization using its training paths only. Choose the six-coefficient version when its score is within strictly 0.1 mm of the best candidate. The three validation paths provide diagnostics; they do not select weights or the candidate.

## Measured inputs and labels

Additional manual labels recover visible ring centers in eight original training episodes and three validation episodes. These are 110 training and 40 validation observations, drawn from the existing recording. Together with the five retained automatic training paths, the final training set contains 176 points: 162 observations and 14 explicitly labeled gap interpolations, across 13 original training episodes. Validation uses 40 observations across three original episodes.

The applicant provided the recording. These centers were visually annotated with assistant support, and native source evidence is included for inspection; this does not claim that the applicant independently labeled or verified every center.

The newly annotated E20/E21/E22 paths are withheld until the model, protocol and physical scene manifest are frozen. They contain 35 observations, including 29 intermediate points and no gap interpolation. These are reserved windows in the same edited recording, with previously visible endpoint configurations; they are not independent recordings. Earlier V3 test diagnostics influenced the redesign. None of these new trajectory labels enters model selection.

Known bearing outer diameter supplies a local pixel-to-metre scale. Image Y is reversed. This approximation treats image displacement as planar displacement; perspective, lifted-object height, tilt and blur can distort it. Reviewed centers have nominal uncertainty of about 5 pixels for clear rims and 8 pixels for blurred rims. Raw proof images, sequential frame ordinals, source PTS and hashes preserve provenance. Sequential decoding corrected the E11/E12 FFmpeg seek ordinal offset before training; V3's historical records remain unchanged. The fresh holdout's 35 ordinals and PTS were independently checked against sequential decoding and ffprobe.

Progress is cumulative observed planar arc length divided by total arc length. It is not elapsed time, depth or robot phase. Source-time progress remains available as distinct evidence. Source-normalized arc-cell quadrature weights observed points by 1 and interpolated points by 0.25, preventing denser sampling from giving an episode extra mass. Endpoint error and intermediate-only error are reported separately.

## Model input and output

At runtime, inputs are the current transfer start XY, the bearing target XY, progress `p` and a camera/task reference angle, default zero. Endpoint direction is normalized to a unit vector. Subtract training-source mean direction and divide by training-source standard deviation, floored at 0.1 per coordinate.

The features are the outer product of `[1, normalized_direction_x, normalized_direction_y]` and `4p(1-p)*[1]` or `4p(1-p)*[1, 2p-1]`. A learned coefficient matrix returns longitudinal and lateral residual in metres. Rotate those two components into the start-to-goal frame, cap the residual norm at 25 mm, and add it to straight interpolation. Start and end are exact anchors; the simulated end is the bearing-hole center, not an imperfect human release offset.

The ridge fit solves a small weighted linear system on CPU. The selected family has six or twelve learned coefficients, with fixed feature statistics estimated from training paths. It is an intentionally small data-driven trajectory model. This design does not require a pretrained vision-language-action model.

Translation invariance follows from relative geometry. Rotating the scene and its reference frame together rotates the resulting curve exactly. Changing start-to-goal direction while keeping the camera reference fixed can change the predicted shape: coverage of new directions therefore requires empirical evaluation, not a claim of unconditional rotational invariance. Distances outside the measured range are extrapolation even though the physical residual cap remains fixed.

## Physical execution and comparison

Generate 65 reference points at transfer entry. Reparameterize this predicted curve by its own arc length, advance the cursor by at most 2 mm per 50 ms control step, and pause cursor advance whenever tool-to-setpoint lag exceeds 4 mm. Cartesian command displacement is also norm-limited to 2 mm. The tool follows through unchanged IK, servos and contact physics. Object state is set only during reset; the ring is never attached, welded or teleported during a rollout.

A V4-specific stage guard also requires reference completion before transfer can become lowering. A legal curved path can pass close to the goal halfway through its arc; proximity alone would skip the remaining curve. `GeneralizationEnv` vetoes that early stage transition after the unchanged base physics step, restoring only the transfer phase counter and observation metadata. It changes no physical state. All three comparisons use this same guard, whose source is bound in the execution signature. A real contact-physics regression checks the short-distance curve that exposed this issue. The historical `sim.py` and V3 checkpoint remain intact.

Approach, grasp, vertical lift, lowering, release, retraction, tool orientation and gripper commands remain geometric. Object positions at runtime come from simulator state. Model inference changes the XY transfer reference; it does not learn perception or all manipulation stages. The hybrid controller's overall task success can consequently be high even if learned human-path fidelity is weak.

All three comparisons use this same new executor and unchanged physics: V4, zero-residual straight reference and the saved V3 profile. V3's result here is explicitly a V3-profile comparison under V4 execution, not its historical 48-case result. Inference counters distinguish 63 evaluated interior reference points from transfer control steps and planned stages.

## Frozen tests

Ninety-six procedural configurations cover three distance bands (45–75, 75–110, 110–160 mm), eight angular bins across the circle and four positions per cell, seed 20261020. Bearing positions lie within x=[0.40,0.52] m, y=[-0.10,0.10] m. The generator conditions geometric positions so the ring starts within x=[0.32,0.62] m, y=[-0.23,0.23] m. This is a declared workspace constraint, not an outcome-based filter. Procedural cases have no human-source ancestry. Every attempted scene stays in the denominator.

The 13 measured training endpoint distances span 66.209–201.077 mm; validation spans 53.486–92.262 mm. Thus the procedural range includes short-distance extrapolation below 66.2 mm, plus novel geometry combinations within the measured distance range. It does not establish long-distance extrapolation.

Success requires the full ring projection inside the bearing hole with 0.3 mm margin, tilt at most 8 degrees, table height within 1.5 mm, linear speed below 0.01 m/s, angular speed below 0.15 rad/s, open fingers, no finger contact and at least 0.4 seconds settling. Reports retain all failures, stage, contacts, physical paths and command paths.

Fresh human-path evaluation compares all models at the same measured arc progress, then computes source-uniform whole-path and intermediate-only displacement. Predeclared sensitivity perturbs intermediate coordinates uniformly in a 5-pixel or 8-pixel square, holds measured endpoints fixed and recomputes arc progress: 100 draws per bound, seed 20261021. This describes measurement sensitivity, not a statistical confidence interval or extra demonstrations. Weights stay frozen throughout.

Demonstration videos replay preset case indices 0,16,32,48,64,80, retaining failures. They are already-counted benchmark cases. Saved evidence and results are linked from the repository README.

## Limits on any conclusion

Same-object motion generalization in this simulator does not establish robustness to new cameras, backgrounds, occlusions, object sizes, clutter, friction or real hardware. Optical path prediction is evaluated against approximate labels, not calibrated 3D pose. A benefit over a straight baseline must be measured independently for path fidelity and task success; equal success cannot support a claim that learned curvature improves placement.
