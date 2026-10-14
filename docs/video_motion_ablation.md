# What does the video-derived curve change?

The V5 black-image and instruction controls show that the model uses RGB and text. They do not isolate the contribution of the recorded transfer motion. This comparison removes that curvature while keeping the rest of the policy fixed.

The straight condition retains the video-derived endpoint geometry. It is a partial motion ablation, not a model trained without real data.

## Intervention

Both models start from the delivered 33,930-parameter V5 checkpoint. The straight readout is fitted only on the 208 TRAIN scenes, with the original source-uniform least squares and nine training phrases. Only the 300 final-layer coefficient rows for intermediate transfer XY are replaced. All other rows, visual/text features, camera buffers, grasp/endpoints, Z and grip keep their original bytes.

The new fit has no epochs or validation-based selection. It inherits V5's selected visual model. The training-only straight residual is below 0.001 mm; this is a model-fit diagnostic, not a real-camera precision claim.

## Fixed evaluation

Seed20261027 fixes 96 new configurations: three distance bands, eight transfer directions and four placements per cell. Each scene runs once with the video curve and once with straight transfer. All192 attempts, including exceptions and timeouts, remain in the denominator. CPU Torch2.14 and two threads are used for both variants.

The two checkpoints, training data, source files, physical assets, scene list and protocol are frozen before execution. Both controllers see the same initial image and instruction. True object state is used for logging and grading, not action generation.

Path metrics use the executor's pre-call waypoint4..19 window, including the preceding state and final post-action state. They measure actual ring motion. Airborne and finger-contact fractions are logged separately; failed grasps are not filtered away. A stationary path has no arc/chord ratio. Duration means steps×0.05s of simulated time.

The analysis independently checks paired observations/common actions and resamples paths by arc length before comparing shapes. Different-duration paths are not compared by matching step indices. Plot scenes0/32/64 are fixed in advance.

## Reproduce

```powershell
./scripts/run_motion_ablation.ps1 -OutputDirectory outputs/new_motion_ablation -Plots
```

Plots require `requirements-plots.txt`. The script prepares and freezes both variants, then runs the registered schedule. It refuses an existing output directory.

[Protocol](superpowers/specs/2026-10-14-motion-ablation-design.md), [baseline-fit code](../ring_learning/motion_ablation_model.py), [physical runner](../ring_learning/motion_ablation.py), [independent analysis](../ring_learning/motion_ablation_analysis.py), [V5 evidence](v5_baseline.md).
