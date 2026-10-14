# Challenge submission checklist

Reference: [official Humanoid Robot Learning Research Intern challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221), read 2026-10-14. Deadline shown: Friday 2026-10-16 at 23:59 BST. The earlier LinkedIn deadline differs; use the official application page.

| Requirement | Evidence | Status |
|---|---|---|
| Personally collected manipulation data | Bundled phone video and hash, 24 endpoints, V4 13 training paths plus reserved/validation evidence | Implemented |
| Real data drives simulated robot | Real optical XY labels train a 12-coefficient V4 model; predictions change physical Panda transfer | Implemented |
| Simple simulation | Physical MuJoCo Panda, annular objects, shared IK/servo and contact grasp | Implemented |
| Run instructions | README commands and isolated-clone report | Included; tested scope recorded |
| Example outputs | Source overlays, six preset case video, saved model, all 288 physical rollouts and charts | Included |
| Design choices, successes and failures | Hybrid control, approximate XY, 96-scene placement generalization, reserved-path errors still worse than straight, equal-success baseline | Included |
| Public GitHub repository | User chose MingxS-cs, then explicitly deferred publication and requested local version | Deferred by user |

Hand-coded solutions are allowed; Libero, SmolVLA and VLA/world-model approaches are examples. No specific push-block task is mandated. Official acceptance or internship selection is not guaranteed. Creativity and research quality remain subjective review criteria.

The applicant's CV, academic eligibility, London availability and application form have not been verified or submitted; they are separate from the challenge implementation.
