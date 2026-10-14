# TinyVLA V5 clean-source reproduction

## Source revision and dependency environments

V5 source was checked out at Git revision `d7168a25cdae8a0884321c61da344296fabd1c0a` in the separate clone `D:\model\ring_bearing_20261014\clean_reproduction_v3`. Tracked files were clean before checkout, and earlier untracked reproduction outputs were preserved. No production source, sealed artifact, original video or frozen precursor was changed by this verification.

Two existing runtimes were used:

| Purpose | Actual runtime | Isolation |
|---|---|---|
| Unit/behavioral tests and artifact comparisons | Clone `.venv`, Python 3.12.10, Torch 2.14.0+cpu, NumPy 2.5.2, MuJoCo 3.12.0, OpenCV 5.0.0, Pillow 12.3.0, pytest 9.1.1 | `include-system-site-packages = false`; imported packages are inside the clone's virtual environment |
| Full pipeline, CUDA training and its CPU inference | Root `.venv-vla`, Python 3.13.5, Torch 2.11.0+cu128, NumPy 2.5.2, MuJoCo 3.12.0, OpenCV 5.0.0, inherited Pillow 12.1.1, pytest 9.1.1 | `include-system-site-packages = true`; GPU runtime reuses Anaconda packages; `ring_learning` imports from the separate source clone |

The GPU runtime detected **NVIDIA GeForce RTX 5080**, compute capability **12.0**, CUDA **12.8** and cuDNN **91900**. An actual CUDA tensor computation succeeded. The same CUDA runtime/version was used as in the delivered model's training; CUDA training was not replaced with CPU training.

This is a clean **source-clone** reproduction on the same Windows host, not a fully isolated GPU dependency installation or a new-machine verification. The isolated CPU environment passed `pip check`. The inherited GPU environment did not: existing NumPy requirements of numba/opencv-python, attrs requirements of pdflatex, and huggingface-hub requirements of transformers conflict with inherited versions. These packages were not altered, and the GPU runtime's Pillow 12.1.1 differs from the 12.3.0 pin in `requirements-vla.txt`. Actual imported module paths and the failed check are retained as evidence. A fresh isolated CUDA installation from that requirements file was not performed.

## Commands and pipeline scope

```powershell
./.venv/Scripts/python.exe -m pytest -q --junitxml=repro_logs/pytest_vla.xml
./scripts/reproduce_vla.ps1 -OutputDirectory outputs/reproduced_vla_clean -PythonExecutable 'D:/model/ring_bearing_20261014/robot_learning/.venv-vla/Scripts/python.exe' -Device cuda
```

The script regenerates train/validation simulator RGB and action chunks from the included frozen V4 precursor, trains for 240 epochs with batch size 8 and 2,000 decoder-prefit steps, prepares/finalizes a new bound scene manifest and model receipt, runs the complete frozen evaluation on CPU in the GPU runtime, and renders six predetermined demonstration clips. The V4 video-to-motion precursor was verified in the separate V4 reproduction; this V5 script reuses it and does not re-extract/retrain V4 from raw phone video.

The phone data contributes through V4's learned planar transfer geometry. V5 inputs are rendered initial simulator RGB and registered text phrases, not the original phone pixels. Height/gripper stages, side-task labels and robot action supervision are synthetic. The model is a task-specific hierarchical TinyVLA with 33,930 parameters and finite registered language, not a pretrained foundation VLA or an open-vocabulary model.

## Verified data and training repeatability

- All 247 tests passed with zero failures/errors/skips in 16.72 seconds.
- Original video, endpoint/motion JSON, development annotations, frozen V4 policy, V5 train/validation NPZ files, dataset manifest, delivered V5 policy and scene/model receipts equal their committed Git blob bytes.
- Regenerated **208 training** and **24 validation** NPZ samples are byte-identical to delivery. Every RGB image, mask, coordinate array, 3 × 23 × 4 task/action chunk, source ID and configuration ID has identical values and raw array bytes. The dataset manifest also has identical bytes.
- Re-training ran all **240 epochs** and selected **epoch 146** by the same validation criterion. The full validation history, decoder training coefficients, selected model configuration and all state tensors have identical values and bytes. Maximum absolute tensor difference is **0**.
- The delivered checkpoint SHA-256 is `412d3f45dafdd47630000c61527c4056e1d8975c01a1a30fc931ec6810dc4452`; the newly trained checkpoint SHA-256 is `7a1ce33d2a127f5191923e47140c0da5bd74579d6962d78170f959c5cdcb4784`. The only changed checkpoint metadata field is `wall_time_seconds`: 58.16 seconds originally versus 68.93 seconds here. Model weights, buffers and all numerical training diagnostics are unchanged.
- All **96 complete scene records** match delivery. Manifest differences are confined to dataset-file paths and the dataset-manifest path in the new output directory.

## Complete execution and evaluation results

The full script exited **0 in 412.86 seconds**. It retained its full 264-attempt schedule; no stage or failing attempt was removed. Inference in this script uses **CPU Torch 2.11.0+cu128** through the shared GPU runtime, whereas the delivered original evaluation used the isolated **CPU Torch 2.14.0+cpu**, Python 3.12.10 runtime. Both use two Torch CPU threads.

| Suite | Actual reproduced outcome |
|---|---|
| Primary ring-inside task | **96/96** successful |
| Black initial image, same scenes | **0/96** successful; all 96 controllers completed and failed the target score |
| Paired registered-language commands | **72/72** successful across 24 scenes × inside/left/right |
| Forbidden controller access attempts | **0** |
| All success/target-score flags and step counts versus delivery | Identical for all **264** attempts |
| Initial RGB hashes versus delivery | Identical for all **264** attempts |

The full summary, including distance/direction stratification and denominator records, equals the delivered summary. The 24 inside-language trials repeat primary scenes; they are not extra independent successes. The benchmark uses 96 fixed-camera synthetic configurations of the same objects and workspace, not 264 independent real recordings.

### Numerical differences across CPU runtimes

The complete per-step rollout dictionaries are **not bit identical across the two CPU runtimes**. Every case has small differences in its predicted chunk, recorded trajectory and physical `info`; other record fields are identical. The maximum state-tensor difference remains zero, and all RGB input hashes remain equal, so this is not a different trained model or different observation dataset.

| Quantity, across all 264 attempts | Observed difference |
|---|---:|
| Predicted XYZ chunk coordinates, maximum absolute component difference | 0.0005066 mm |
| Predicted XYZ chunk coordinates, mean absolute component difference | 0.00006234 mm |
| Predicted gripper scalar, maximum absolute difference | 1.192 × 10⁻⁷ |
| Executed gripper action, maximum absolute difference | 5.960 × 10⁻⁸ |
| Gripper open/close action counts | Identical in every case |
| Controller steps and trajectory lengths | Identical in every case |
| Ring coordinate components during physical execution, maximum absolute difference | 0.3469 mm |
| Tool coordinate components during physical execution, maximum absolute difference | 0.1245 mm |
| Final ring position, maximum Euclidean difference | 0.06440 mm |
| Final ring position, mean Euclidean difference | 0.0008199 mm |
| Success and target-score flags | Identical in every case |

Paired diagnostic counts are unchanged. Its only numerical differences are the mean maximum left/right waypoint changes: absolute differences of `2.409021059648908e-7 m` and `5.960464477539063e-8 m`. These values are retained rather than hidden behind an equality claim.

To isolate the runtime difference, the first matched scene was replayed with the **same newly trained checkpoint**, same RGB hash and two threads in the clone's Torch 2.14.0+cpu environment. Its **entire rollout dictionary**, including predicted chunk, every trajectory state/action, physical info and counters, exactly equals the delivered original record. The Torch 2.11 result for that scene differs. This establishes a same-version exact match for that representative scene and explains the cross-runtime numerical boundary; it does not assert a second same-version replay of all 264 cases. That unnecessary full repeat was not performed after the runtime difference was resolved.

## Demonstration decoding and optional analysis

All six preset demonstrations succeeded. Their task/index, success, steps, configuration IDs and neural-forward counters equal delivery. Predicted chunks and physical info retain the same cross-runtime numerical differences described above; full case metadata is not claimed bit identical.

The six clips were decoded to end, with frame counts **152, 167, 183, 163, 135, 145**. The concatenated video also decoded fully: **945 frames**, **94.5 seconds**, **10 fps**, **840 × 414 pixels**. These clips repeat already-evaluated scenes and are not additional benchmark trials.

The optional `ring_learning.vla_analysis --base outputs/reproduced_vla_clean` command exited 0 using the isolated CPU environment, produced its plots and verification JSON, and reported all 264 attempts completed with zero forbidden accesses. Its primary final ring-center error was 0.507 mm on average and 1.123 mm at maximum; these are simulator target errors from the reproduced Torch 2.11 rollout, not calibrated real-camera accuracy.

## Evidence and validated scope

Compact evidence is in `outputs/reproduction/vla_final`: runtime/module paths for both environments, inherited dependency conflicts, tests/JUnit results, exact dataset/model/scene comparisons, complete-evaluation comparison summaries, numerical difference bounds, representative Torch 2.14 scene proof, training history, regenerated freeze receipt and fully decoded video checks. The large complete evaluation and newly generated video remain in the separate clone at `outputs/reproduced_vla_clean`, so they were not duplicated into the evidence folder.

The model uses one initial simulator RGB image and finite registered language to predict a 23-waypoint XYZ/gripper plan, then a hand-written waypoint schedule/servo executes it with own proprioception. It has no validated real-camera transfer, open vocabulary, unknown-object handling or visual replanning during execution. True object state is available to logging/grading, with zero forbidden object-state accesses by the action controller in these runs. The original data and the entire project remain local; no public repository or application was submitted by this reproduction.
