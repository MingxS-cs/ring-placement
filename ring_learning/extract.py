"""Extract visible annular endpoints; metric coordinates use an approximate local scale.

Run ``python -m ring_learning.extract --help``. The source movie is only read.
Dark-ring footprint centers are image evidence, not recovered table contact or 6D poses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


DIMENSIONS_M = {
    "bearing_outer_diameter": 0.050,
    "bearing_inner_diameter": 0.040,
    "bearing_height": 0.005,
    "ring_outer_diameter": 0.025,
    "ring_inner_diameter": 0.020,
    "ring_height": 0.007,
}
MAPPING_DISCLAIMER = (
    "Approximate known-diameter normalization of image-plane coordinates, not calibrated "
    "depth, camera intrinsics, metric pose ground truth or 6D recovery. A single local "
    "isotropic scale ignores perspective, lens distortion, object-height parallax and "
    "nonuniform pixel scale across the tabletop. World axes and origin are synthetic."
)


def _ellipse_candidates(mask, minimum_diameter, maximum_diameter, circularity=0.60):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        if len(contour) < 30:
            continue
        (u, v), (a, b), angle = cv2.fitEllipse(contour)
        diameter = math.sqrt(a * b)
        perimeter = cv2.arcLength(contour, True)
        circ = 4 * math.pi * cv2.contourArea(contour) / max(perimeter**2, 1e-8)
        if not (minimum_diameter <= diameter <= maximum_diameter):
            continue
        if min(a, b) / max(a, b) < 0.60 or circ < circularity:
            continue
        candidates.append({
            "center_px": [float(u), float(v)],
            "outer_diameter_px": float(diameter),
            "ellipse_axes_px": [float(a), float(b)],
            "ellipse_angle_deg": float(angle),
            "circularity": float(circ),
        })
    return candidates


def detect_scene(image: np.ndarray, terminal=False) -> dict:
    """Detect a bright concentric bearing rim and a bounded dark annular footprint.

    Bounds target the 720x1280 source image and scale with image width. Missing or
    ambiguous annuli remain explicit; the detector never supplies hidden coordinates.
    """
    if image is None or image.ndim != 3:
        raise ValueError("Expected a BGR image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width_scale = image.shape[1] / 720
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 40, 100)
    rims = _ellipse_candidates(edges, 135 * width_scale, 230 * width_scale, 0.72)
    bearing_options = []
    for rim in rims:
        center = np.asarray(rim["center_px"])
        diameter = rim["outer_diameter_px"]
        inner_rims = [other for other in rims if
                      np.linalg.norm(center - other["center_px"]) < 0.04 * diameter
                      and 0.72 < other["outer_diameter_px"] / diameter < 0.92]
        if inner_rims:
            bearing_options.append(rim)
    bearing = max(bearing_options, key=lambda rim: rim["outer_diameter_px"], default=None)
    bearing_method = "Canny contours; concentric closed ellipse rims; largest supported rim"
    if bearing is None:
        # The metal is less saturated than the beige table; this recovers complete
        # visible rims whose grayscale edges join the bearing's own shadow.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        metal = cv2.inRange(hsv, (0, 0, 130), (179, 40, 255))
        metal = cv2.morphologyEx(metal, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        metal_rims = _ellipse_candidates(metal, 135 * width_scale, 230 * width_scale, 0.65)
        for rim in metal_rims:
            if any(np.linalg.norm(np.asarray(rim["center_px"]) - inner["center_px"])
                   < 0.04 * rim["outer_diameter_px"]
                   and 0.72 < inner["outer_diameter_px"] / rim["outer_diameter_px"] < 0.92
                   for inner in metal_rims):
                bearing_options.append(rim)
        bearing = max(bearing_options, key=lambda rim: rim["outer_diameter_px"], default=None)
        bearing_method = "Low-saturation bright metal mask; concentric closed ellipse rims"
    if bearing is None:
        return {"status": "missing_object", "bearing": None, "ring": None,
                "reason": "No bounded concentric bearing rims"}
    bearing["method"] = bearing_method
    bearing["confidence"] = min(0.95, bearing["circularity"])
    diameter = bearing["outer_diameter_px"]
    center = np.asarray(bearing["center_px"])
    ring_options = []
    apertures = []
    for threshold in (65, 80, 95, 110, 125):
        mask = cv2.inRange(gray, 0, threshold)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        small = _ellipse_candidates(mask, 0.22 * diameter, 0.69 * diameter)
        if terminal:
            for aperture in small:
                if (0.25 < aperture["outer_diameter_px"] / diameter < 0.43
                        and np.linalg.norm(center - aperture["center_px"]) < 0.25 * diameter):
                    apertures.append((aperture, threshold))
        for outer in small:
            ratio = outer["outer_diameter_px"] / diameter
            if not 0.43 < ratio < 0.66:
                continue
            separation = np.linalg.norm(center - outer["center_px"])
            if terminal and separation > 0.30 * diameter:
                continue
            if not terminal and separation < 0.60 * diameter:
                continue
            holes = [inner for inner in small if
                     np.linalg.norm(np.asarray(outer["center_px"]) - inner["center_px"])
                     < 0.14 * outer["outer_diameter_px"]
                     and 0.45 < inner["outer_diameter_px"] / outer["outer_diameter_px"] < 0.90]
            if not holes:
                continue
            outer["threshold_gray"] = threshold
            outer["method"] = "Dark mask; 5px closure; nested circular ellipse contours"
            outer["diameter_kind"] = "outer_optical_footprint"
            outer["confidence"] = min(0.90, outer["circularity"])
            outer["score"] = outer["circularity"] - 0.3 * abs(ratio - 0.53)
            ring_options.append(outer)
    ring = max(ring_options, key=lambda option: option["score"], default=None)
    if ring is None and terminal:
        # When the outer wall joins a shadow, an observed closed aperture and dark
        # surrounding wall still identify its optical center. Do not infer a rim.
        valid_apertures = []
        for aperture, threshold in apertures:
            u, v = aperture["center_px"]
            a, b = np.asarray(aperture["ellipse_axes_px"]) * 0.5 * 1.20
            angle = math.radians(aperture["ellipse_angle_deg"])
            theta = np.linspace(0, 2 * math.pi, 72, endpoint=False)
            xs = u + a*np.cos(theta)*math.cos(angle) - b*np.sin(theta)*math.sin(angle)
            ys = v + a*np.cos(theta)*math.sin(angle) + b*np.sin(theta)*math.cos(angle)
            samples = gray[np.clip(np.rint(ys).astype(int), 0, gray.shape[0]-1),
                           np.clip(np.rint(xs).astype(int), 0, gray.shape[1]-1)]
            if np.mean(samples < max(125, threshold)) >= 0.75:
                aperture["threshold_gray"] = threshold
                aperture["method"] = "Visible closed inner aperture ellipse with surrounding dark wall"
                aperture["diameter_kind"] = "inner_aperture_not_outer_rim"
                aperture["confidence"] = 0.60
                aperture["score"] = aperture["circularity"]
                valid_apertures.append(aperture)
        ring = max(valid_apertures, key=lambda option: option["score"], default=None)
    if ring is None:
        return {"status": "missing_object", "bearing": bearing, "ring": None,
                "reason": "No bounded dark annulus with visible hole"}
    ring.pop("score")
    return {"status": "detected", "bearing": bearing, "ring": ring}


def map_to_world(bearing_center_px, bearing_outer_diameter_px, ring_center_px) -> dict:
    """Apply the documented image axes and local 50 mm bearing diameter scale."""
    coordinates = np.asarray([bearing_center_px, ring_center_px], dtype=float)
    if (coordinates.shape != (2, 2) or not np.isfinite(coordinates).all()
            or not np.isfinite(bearing_outer_diameter_px) or bearing_outer_diameter_px <= 0):
        raise ValueError("Finite 2D centers and a positive bearing diameter are required")
    (bu, bv), (ru, rv) = coordinates
    scale = DIMENSIONS_M["bearing_outer_diameter"] / bearing_outer_diameter_px
    bearing = np.array([0.45 + (bu - 360) * scale, (640 - bv) * scale])
    ring = bearing + np.array([ru - bu, bv - rv]) * scale
    return {"bearing_xy_m": bearing.tolist(), "ring_xy_m": ring.tolist(),
            "meters_per_pixel": float(scale)}


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Cannot read source frame {frame_index}")
    # OpenCV's FFmpeg backend reports the decoded source PTS after each read.
    return frame, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000


def resolve_endpoint_frames(candidate, next_start, reviewed_selection=None):
    """Apply recorded visible-frame selection while enforcing the candidate window."""
    selection = reviewed_selection or {}
    first = candidate["source_frame_start"]
    initial = selection.get("initial_frame", first)
    terminal = selection.get("terminal_frame", candidate["terminal_evidence_frame"])
    if not first <= initial < terminal < next_start:
        raise ValueError("Selected endpoint crosses the candidate window")
    return int(initial), int(terminal)


def _annotate(image, detection, label):
    result = image.copy()
    for name, color in (("bearing", (0, 200, 0)), ("ring", (255, 80, 20))):
        item = detection.get(name)
        if item is None:
            continue
        u, v = item["center_px"]
        a, b = item["ellipse_axes_px"]
        cv2.ellipse(result, ((u, v), (a, b), item["ellipse_angle_deg"]), color, 2)
        cv2.drawMarker(result, (round(u), round(v)), color, cv2.MARKER_CROSS, 15, 2)
        cv2.putText(result, f"{name} ({u:.1f},{v:.1f})", (max(0, round(u)-80), round(v)-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(result, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 230), 2)
    return result


def extract_video(source, candidates_path, output_json, output_dir, visual_review_path=None):
    source, candidates_path = Path(source), Path(candidates_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(candidates_path.read_text(encoding="utf-8"))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != metadata["source_sha256"]:
        raise ValueError("Source SHA-256 differs from inspected video")
    visual_review_path = Path(visual_review_path or output_dir / "visual_review.json")
    visual_review = {}
    if visual_review_path.exists():
        visual_review = json.loads(visual_review_path.read_text(encoding="utf-8"))
        if visual_review.get("source_sha256") != sha:
            raise ValueError("Visual review is for a different source video")
    reviewed_episodes = visual_review.get("episodes", {})
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {source}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    episodes, rejected, review = [], [], []
    candidates = metadata["candidates"]
    for index, candidate in enumerate(candidates):
        episode_id = candidate["candidate_id"]
        next_start = candidates[index+1]["source_frame_start"] if index+1 < len(candidates) else frame_count
        episode_review = reviewed_episodes.get(episode_id, {})
        start_index, terminal_index = resolve_endpoint_frames(candidate, next_start, episode_review)
        start_image, start_pts = read_frame(cap, start_index)
        terminal_image, terminal_pts = read_frame(cap, terminal_index)
        initial = detect_scene(start_image)
        terminal = detect_scene(terminal_image, terminal=True)
        images = []
        for phase, image, detection, frame_index in (("initial", start_image, initial, start_index),
                                                     ("terminal", terminal_image, terminal, terminal_index)):
            name = f"{episode_id}_{phase}.jpg"
            annotated = _annotate(image, detection, f"{episode_id} {phase} f{frame_index} {detection['status']}")
            cv2.imwrite(str(output_dir / name), annotated)
            images.append(name)
        review.append({"episode_id": episode_id, "initial": initial, "terminal": terminal,
                       "source_frame_start": start_index, "source_frame_terminal": terminal_index,
                       "images": images})
        if (initial["status"] != "detected" or terminal["status"] != "detected"
                or episode_review.get("decision") == "rejected"):
            rejected.append({"episode_id": episode_id, "reason": episode_review.get("notes", "Missing visible endpoint detection"),
                             "initial_status": initial["status"], "terminal_status": terminal["status"]})
            continue
        bearing = initial["bearing"]
        mapped = map_to_world(bearing["center_px"], bearing["outer_diameter_px"], initial["ring"]["center_px"])
        final_bearing = terminal["bearing"]
        final_map = map_to_world(final_bearing["center_px"], final_bearing["outer_diameter_px"], terminal["ring"]["center_px"])
        final_offset = np.asarray(final_map["ring_xy_m"]) - final_map["bearing_xy_m"]
        episodes.append({
            "episode_id": episode_id,
            "scene_group": candidate["scene_group_candidate"],
            "source_frame_start": start_index,
            "source_frame_terminal": terminal_index,
            "candidate_original_start_frame": candidate["source_frame_start"],
            "candidate_original_terminal_frame": candidate["terminal_evidence_frame"],
            "source_pts_start_s": start_pts,
            "source_pts_terminal_s": terminal_pts,
            "candidate_window_end_exclusive_frame": next_start,
            "bearing_center_px": bearing["center_px"],
            "bearing_outer_diameter_px": bearing["outer_diameter_px"],
            "terminal_bearing_center_px": final_bearing["center_px"],
            "terminal_bearing_outer_diameter_px": final_bearing["outer_diameter_px"],
            "ring_start_center_px": initial["ring"]["center_px"],
            "ring_final_center_px": terminal["ring"]["center_px"],
            "bearing_xy_m": mapped["bearing_xy_m"],
            "ring_xy_m": mapped["ring_xy_m"],
            "final_offset_xy_m": final_offset.tolist(),
            "duration_s": terminal_pts - start_pts,
            "detection": {"initial": initial, "terminal": terminal,
                          "provenance": ("Automatic contour extraction; manual visual overlay review"
                                         if episode_review else "Automatic contour extraction; pending visual review"),
                          "confidence": min(initial["bearing"]["confidence"], initial["ring"]["confidence"],
                                            terminal["bearing"]["confidence"], terminal["ring"]["confidence"]),
                          "confidence_definition": "Heuristic contour fit/evidence score, not a calibrated probability or metric accuracy estimate",
                          "manual_correction": start_index != candidate["source_frame_start"] or terminal_index != candidate["terminal_evidence_frame"],
                          "manual_correction_type": "visible frame selection" if start_index != candidate["source_frame_start"] or terminal_index != candidate["terminal_evidence_frame"] else None,
                          "visually_reviewed": bool(episode_review),
                          "review_notes": episode_review.get("notes"),
                          "pixel_uncertainty_note": "Centers are approximate visible optical footprints; wall height and shadows can shift centers by about 5-12 pixels. This is a qualitative estimate, not a measured error bound."},
            "evidence_images": images,
            "coordinate_source": "visible optical footprint; known-diameter normalization",
            "release_time_exact_s": None,
            "ground_contact_verified": False,
            "continuous_stability_verified": False,
        })
    cap.release()
    for offset in range(0, len(review), 6):
        sheet = Image.new("RGB", (1440, 900), "white")
        draw = ImageDraw.Draw(sheet)
        for column, item in enumerate(review[offset:offset+6]):
            for row, filename in enumerate(item["images"]):
                image = Image.open(output_dir / filename)
                image.thumbnail((240, 425))
                sheet.paste(image, (column*240, row*450+25))
                draw.text((column*240+3, row*450+5), filename, fill="black")
        sheet.save(output_dir / f"contact_sheet_{offset//6+1}.jpg")
    result = {
        "schema_version": 1,
        "task_spec": {"dimensions_m": DIMENSIONS_M},
        "source": {"basename": source.name, "sha256": sha,
                   "frame_count": frame_count,
                   "image_dimensions_px": image_size,
                   "pts_provenance": "OpenCV FFmpeg CAP_PROP_POS_MSEC after decoded frame"},
        "mapping": {"method": "local bearing diameter normalization",
                    "disclaimer": MAPPING_DISCLAIMER,
                    "diameter_definition": "geometric mean of visible bearing ellipse axes",
                    "formula": "s=0.05/D; bearing=[0.45+(bu-360)*s,(640-bv)*s]; ring=bearing+[(ru-bu),(bv-rv)]*s",
                    "terminal_offset": "Terminal ring minus terminal bearing with terminal local scale; excludes camera shake"},
        "boundary_policy": "Bounded candidate endpoints only; no velocity/action differences across edited cuts",
        "episodes": episodes,
        "rejected_candidates": rejected,
        "review_status": ("all candidate endpoint overlays visually reviewed"
                          if set(reviewed_episodes) == {candidate["candidate_id"] for candidate in candidates}
                          else "pending visual contact-sheet review"),
        "visual_review_file": str(visual_review_path) if visual_review else None,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "detection_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=r"C:\Users\minso\Downloads\359977334.mp4")
    parser.add_argument("--candidates", default="../inspection_359977334/episode_candidates.json")
    parser.add_argument("--output", default="data/real_episodes.json")
    parser.add_argument("--output-dir", default="outputs/extraction")
    parser.add_argument("--visual-review", default=None)
    args = parser.parse_args()
    result = extract_video(args.source, args.candidates, args.output, args.output_dir, args.visual_review)
    print(json.dumps({"accepted": len(result["episodes"]), "rejected": len(result["rejected_candidates"]),
                      "review_status": result["review_status"]}))


if __name__ == "__main__":
    main()
