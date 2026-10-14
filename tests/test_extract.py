"""Behavioral checks for dimension-normalized visible endpoint extraction."""
import importlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


def extractor():
    spec = importlib.util.find_spec("ring_learning.extract")
    assert spec is not None, "Video endpoint extractor has not been implemented"
    return importlib.import_module("ring_learning.extract")


def scene(include_bearing=True, include_ring=True):
    image = np.full((1280, 720, 3), (170, 193, 212), dtype=np.uint8)
    if include_bearing:
        cv2.circle(image, (360, 640), 90, (200, 200, 200), -1)
        cv2.circle(image, (360, 640), 90, (95, 95, 95), 2)
        cv2.circle(image, (360, 640), 72, (170, 193, 212), -1)
        cv2.circle(image, (360, 640), 72, (95, 95, 95), 2)
    if include_ring:
        cv2.circle(image, (170, 520), 45, (30, 30, 30), -1)
        cv2.circle(image, (170, 520), 36, (170, 193, 212), -1)
    return image


def test_detects_separate_annuli_and_preserves_center_separation():
    result = extractor().detect_scene(scene())
    assert result["status"] == "detected"
    assert result["bearing"]["center_px"] == pytest.approx([360, 640], abs=3)
    assert result["bearing"]["outer_diameter_px"] == pytest.approx(180, abs=5)
    assert result["ring"]["center_px"] == pytest.approx([170, 520], abs=3)
    mapped = extractor().map_to_world(
        result["bearing"]["center_px"],
        result["bearing"]["outer_diameter_px"],
        result["ring"]["center_px"],
    )
    assert np.subtract(mapped["ring_xy_m"], mapped["bearing_xy_m"]) == pytest.approx(
        [-190 * 0.05 / 180, 120 * 0.05 / 180], abs=0.002
    )


@pytest.mark.parametrize("bearing,ring", [(False, True), (True, False), (False, False)])
def test_rejects_missing_objects(bearing, ring):
    result = extractor().detect_scene(scene(bearing, ring))
    assert result["status"] == "missing_object"
    assert result["bearing"] is None or result["ring"] is None


def test_mapping_is_explicit_and_rejects_invalid_scale():
    result = extractor().map_to_world([360, 640], 200, [460, 540])
    assert result["bearing_xy_m"] == pytest.approx([0.45, 0.0])
    assert result["ring_xy_m"] == pytest.approx([0.475, 0.025])
    assert result["meters_per_pixel"] == pytest.approx(0.00025)
    with pytest.raises(ValueError):
        extractor().map_to_world([360, 640], 0, [460, 540])


def test_terminal_ring_inside_bearing_is_detected():
    image = scene(include_ring=False)
    cv2.circle(image, (365, 646), 45, (30, 30, 30), -1)
    cv2.circle(image, (365, 646), 36, (170, 193, 212), -1)
    result = extractor().detect_scene(image, terminal=True)
    assert result["status"] == "detected"
    assert result["ring"]["center_px"] == pytest.approx([365, 646], abs=3)


@pytest.mark.parametrize("frame,terminal", [(360, True), (2970, False), (3222, True), (2190, False), (2310, True)])
def test_real_low_contrast_rims_are_recovered(frame, terminal):
    source = Path(r"C:\Users\minso\Downloads\359977334.mp4")
    if not source.exists():
        pytest.skip("Original source movie is unavailable")
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    assert ok
    result = extractor().detect_scene(image, terminal=terminal)
    assert result["status"] == "detected"
    assert 160 < result["bearing"]["outer_diameter_px"] < 200


def test_reviewed_frame_selection_stays_inside_episode():
    module = extractor()
    assert hasattr(module, "resolve_endpoint_frames"), "Reviewed frame selection is missing"
    candidate = {"source_frame_start": 100, "terminal_evidence_frame": 180}
    assert module.resolve_endpoint_frames(candidate, 210, {"terminal_frame": 190}) == (100, 190)
    with pytest.raises(ValueError, match="window"):
        module.resolve_endpoint_frames(candidate, 210, {"terminal_frame": 211})
    with pytest.raises(ValueError, match="window"):
        module.resolve_endpoint_frames(candidate, 210, {"initial_frame": 99})

