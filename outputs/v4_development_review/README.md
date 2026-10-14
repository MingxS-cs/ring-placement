# Development motion annotation review

`data/v4_development_annotations.json` is frozen at SHA-256 `65fcebde93e73e94d37b7bb41cabae12e4c930c7525797b89ae61634b71ca0e6`.

This review adds 110 genuine visible samples in eight original training episodes (E03, E06, E07, E13, E15, E17, E18, E19) and 40 in all three original validation episodes (E05, E11, E12). It does not change original split membership. All 150 samples are `manual_visual`; no interpolated or hidden coordinates count as observations. The existing five reliable training episodes remain separate existing evidence.

Source: bundled `data/source/359977334.mp4`, SHA-256 `90cfbb066d2d88c1794ee7f7d517715f50a72dda8abf7248624650d0d4ca0dd4`. Only assigned development episodes were reviewed. No fresh holdout or old test episodes were accessed or annotated.

Each `*_center_review.png` shows native-sized 200-pixel crops with a center cross and a nominal five-pixel radius marker. `native_frames/` contains the exact untouched 720×1280 source sample PNGs. `native_overlays/` contains full-size overlays. JSON `decoded_rgb_sha256` verifies exact source PNG pixels, while frame ordinal and actual source PTS were checked independently against sequential decoding. Overviews and transfer sheets are viewing aids.

The source contains a PTS offset; frame ordinal divided by nominal FPS is not reliable. FFmpeg seeking also returned ordinal n−1 for every reviewed E11/E12 image requested as n. `ordinal_lineage.json` records exact pixel-hash and PTS matches between the initial seeked review images and sequentially decoded frames. Final JSON and native proof use the corrected true ordinals (odd frames for E11/E12), and those two overview/transfer sheets were regenerated. Provisional E11/E12 JPEG caches are removed because they do not provide authoritative frame lineage.

Centers were selected from the visible ring rim/aperture, excluding the displaced cast shadow. E03 specifically corrects the shadow association in the original automatic f428–434 segment. E05 f730/f732/f734 now have genuine visible evidence. Qualitative optical center uncertainty is approximately ±5 px, widened to ±8 px where the rim is visibly blurred; these are not calibrated statistical intervals. A visible sidewall, motion blur, and optical perspective limit geometric accuracy.

Transfer samples begin near pre-lift and end at a clearly visible ring inside the bearing hole. Sampling is dense through transfer; the final release confirmation can follow a longer stationary interval. No hidden intermediate coordinates are fabricated. `path`/`derived_path` normalize the sparse genuine observed XY polyline by arc length. Such downstream polyline interpolation is a representation of the observed path, not additional observations.

Metric XY uses the original episode bearing center and visible outer diameter with the known 50 mm physical diameter. This is approximate image-plane normalization, without measured depth, camera calibration, physical height, orientation, or 3D ground truth. The verification report's terminal offsets and arc lengths are therefore approximate optical quantities.

`verification.json` records successful checks for source hash, exact sequential source ordinal/PTS/pixels, native PNG hashes, original split membership, in-episode bounds, manual sample kinds, path reconstruction, and terminal approximate offsets within the 7.5 mm radial ring-to-hole clearance.

The review scripts are artifact-generation utilities only. No production `ring_learning` source or checkpoint assets were modified.
