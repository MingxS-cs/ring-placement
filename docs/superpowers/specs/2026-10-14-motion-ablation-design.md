# Video-motion ablation and submission cleanup

The user approved a matched video-curve/straight-transfer experiment and improvements to both code and prose. Publication remains deferred. No AI-detection score is an acceptance criterion; assistance stays disclosed.

## Question

Which robot actions change because of the recorded transfer motion? The existing black-image and language controls answer different questions. This experiment removes the video-derived curvature while retaining the recorded endpoint geometry.

## Controlled intervention

- Use the delivered V5 checkpoint, RGB encoder, word embeddings, hidden text features, camera, physics and waypoint executor.
- Copy the original training action arrays. Replace only transfer XY at waypoints4..18 with linear interpolation from the lift position (waypoint3) to the transfer endpoint (waypoint19), at progress1/16..15/16.
- Fit the straight coefficient targets using the existing source-uniform training-only least-squares routine and all nine existing training phrases.
- Update only final-layer coefficient rows for those15 XY positions. Every other parameter row, all Z/gripper outputs, grasp/release/endpoints and camera buffers retain the original bytes.
- Use no validation or test labels to select, tune or retry the model. Verify common parameter tensors/rows before freezing either model. The known feature design has rank9, not10.
- This is a partial motion ablation: video-derived source endpoint geometry still remains in both conditions. It does not remove all personally collected data.

## Evaluation

Freeze a new seed20261027 and96 scenes before running either model. Use the same three45–75/75–110/110–160mm distance bands, eight transfer directions and four placements per cell as V5. Draw bearing/ring locations with the same workspace bounds. Keep the original V5 seed and outputs unchanged.

Run each scene twice, with the video-motion and straight-transfer models, in the same CPU Torch2.14 runtime with two threads. Use the existing restricted controller view and physical runner. Keep all192 attempts, including exceptions and timeouts. Success requires controller completion plus the existing settled inside-target score. True object state is for metrics/logging only.

Report paired initial image hashes, common predicted XYZ/gripper outputs, changed transfer paths, actual ring trajectories, success, final center error and controller steps. Physical path length and lateral departure use the executor's transfer window (pre-call waypoint4..19), including its pre-action starting state and final post-action state. Record airborne/contact fractions separately; a failed grasp or dropped ring is retained rather than filtered into an apparently good airborne path. Zero-chord paths have an explicit degenerate flag and no arc/chord ratio. Step counts times0.05s are simulated time, not wall-clock or human timing. These paths are not human3D ground truth. Differences do not establish superiority unless a relevant metric improves. Do not count this reproduction or repeated old cases as new independent evidence.

Bind the two checkpoints, baseline-fit report, dataset manifest/arrays, inherited V5 receipt, implementation files, physical environment, this protocol and the complete scene list before execution. Existing frozen files are immutable. New outputs refuse overwrites.

## Submission prose and code

Keep new code small and reuse the tested V5 physics/controller; do not refactor hash-bound modules. Add focused tests for the counterfactual label intervention, preserved weights, fresh scene coverage, tamper detection, failure denominators and actual-path metrics.

Rewrite the root README and Chinese overview around the concrete task, data origin, commands, actual results and remaining limitations. Retain earlier V5 documentation as a historical baseline. Remove repeated verification slogans and template prose; preserve numeric evidence and failed experiments.

Record assistance honestly: the applicant chose the task, recorded the video and supplied dimensions; AI assisted code, extraction, experiment automation and writing. Do not invent personal debugging experiences, hide AI involvement or promise detector evasion.

## Completion

Focused tests and the existing regression suite pass; all frozen V5 bindings still validate; all192 cases are retained and summarized; independent review checks fairness and claims; figures and any demo decode; local commit and a new archive receive CRC/all-member verification. No public repository or application is submitted.
