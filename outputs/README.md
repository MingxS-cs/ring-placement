# Experiments

Start with the current model and paired motion comparison. Other folders retain the data conversion, earlier policies and failed attempts that led to them.

| Run | Files |
|---|---|
| Visual-language policy | [Model](vla_v5/model/policy.pt), [training data](vla_v5/dataset/manifest.json), [training history](vla_v5/model/training.json) |
| Position, direction and distance benchmark | [264 rollouts including image/language controls](vla_v5/evaluation.json), [preset demonstrations](vla_v5/student_demo.mp4), [frozen model and protocol](vla_v5/model_freeze.json) |
| Video curve versus straight transfer | [192 rollouts](motion_ablation_v6_final/evaluation.json), [path plots](motion_ablation_v6_final/analysis/paired_ring_paths.png), [input/action checks](motion_ablation_v6_final/analysis/verification.json) |
| Source-video processing | [Endpoint review](extraction/visual_review.json), [motion review](motion/visual_review.json) |
| Video-derived motion precursor | [Geometry model](generalization_v4/model/policy.json), [reserved-human-path result](generalization_v4/fidelity/fidelity.json) |
| Automated reproduction | [Policy](reproduction/vla_final/), [motion comparison](reproduction/motion_ablation_v6/corrected_4536823/) |

`model/`, `model_v2/`, `model_dagger/`, `video_model/` and `evaluation_final/` belong to earlier baselines. They are not the current visual-language model. The initial `motion_ablation_v6/` run failed the image-equality control; its complete records and source snapshot remain available alongside the corrected run.

Versioned paths are preserved because models and freeze receipts bind them. Directory names, model weights and old execution records were not rewritten during the presentation cleanup. The history is described in [earlier reports](../docs/archive/README.md).
