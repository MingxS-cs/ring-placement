# Challenge submission checklist

Reference: [official Humanoid Robot Learning Research Intern challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221), read 2026-10-14. Deadline shown: Friday 2026-10-16 at 23:59 BST. The earlier LinkedIn deadline differs; use the official application page.

| Requirement | Evidence | Status |
|---|---|---|
| Personally collected manipulation data | Bundled phone video and hash, 24 endpoints; 13 training-source paths and separate validation/test/reserved evidence. V5 uses 208/24 source-isolated synthetic descendants | Implemented |
| Real data drives simulated robot | Real optical XY labels fit the preserved train-only V4 precursor, which supplies V5 synthetic action curves. The 33,930-parameter V5 takes initial simulator RGB and finite text to produce 23 XYZ/gripper waypoints executed by a physical Panda | Implemented; synthetic vision and action-label scope disclosed |
| Simple simulation | Physical MuJoCo Panda, annular objects, shared IK/servo and contact grasp | Implemented |
| Run instructions | README commands and isolated-clone report | Included; tested scope recorded |
| Example outputs | Source overlays, six preset V5 case videos, frozen checkpoint, all 264 V5 physical attempts and charts; V4 evidence retained | Included |
| Design choices, successes and failures | Hierarchical finite-command VLA; 96/96 main, 0/96 black-image and 72/72 language controls (24 inside repeats); two failed validation designs; fixed robot-hidden camera and manual execution disclosed. V4 negative human-path result and equal-success baseline retained | Included |
| Public GitHub repository | User chose MingxS-cs, then explicitly deferred publication and requested local version | Deferred by user |

Hand-coded solutions are allowed; Libero, SmolVLA and VLA/world-model approaches are examples. No specific push-block task is mandated. Official acceptance or internship selection is not guaranteed. Creativity and research quality remain subjective review criteria.

The applicant's CV, academic eligibility, London availability and application form have not been verified or submitted; they are separate from the challenge implementation.
