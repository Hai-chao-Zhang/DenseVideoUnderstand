"""Deterministic source-only reference correction for High-motion v2.

No model outputs are accepted. Invalid projections have no grid label and are
never clamped into the image or replaced with a center-cell sentinel.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


VERSION = "highmotion-right-ring-v2"
TARGET_JOINT = "rightRingFingerMetacarpal"
MASK_POLICY = "finite-positive-depth-in-frame-positive-confidence-v1"
GRID_NAMES = (
    ("topleft", "top", "topright"),
    ("left", "middle", "right"),
    ("bottomleft", "bottom", "bottomright"),
)


def _dimension(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _transforms(value, name):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4) or not len(array):
        raise ValueError(f"{name} must have shape (T, 4, 4), T > 0")
    return array


def project_reference(intrinsic, camera, joint, confidence, width, height):
    """Project the named joint and return full-length labels, points and masks.

    Camera and joint transforms share a world frame. The camera-space joint
    origin is ``(inverse(camera) @ joint)[:3, 3]``. Positive depth means its
    camera-space Z coordinate is strictly positive. Confidence must be present,
    finite, and in (0, 1]. Shape errors raise; missing or invalid frame values
    are masked. Reasons are pipe-separated, in deterministic evaluation order;
    an empty reason string denotes a valid reference. Confidence is a source
    reliability check, not proof of RGB visibility or perfect reprojection.
    """

    width, height = _dimension(width, "width"), _dimension(height, "height")
    camera = _transforms(camera, "camera")
    joint = _transforms(joint, "joint")
    if camera is None and joint is None:
        raise ValueError("At least one transform array is needed to establish T")
    count = len(camera) if camera is not None else len(joint)
    if camera is not None and joint is not None and camera.shape != joint.shape:
        raise ValueError("Camera and joint frame counts differ")
    intrinsic = None if intrinsic is None else np.asarray(intrinsic, dtype=np.float64)
    if intrinsic is not None and intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must have shape (3, 3)")
    confidence = None if confidence is None else np.asarray(confidence, dtype=np.float64)
    if confidence is not None and confidence.shape != (count,):
        raise ValueError("confidence must have shape (T,)")

    reasons = [[] for _ in range(count)]

    def mark(mask, reason):
        for index in np.flatnonzero(mask):
            reasons[int(index)].append(reason)

    all_frames = np.ones(count, dtype=bool)
    if confidence is None:
        mark(all_frames, "missing_confidence")
    else:
        mark(~np.isfinite(confidence) | (confidence <= 0) | (confidence > 1),
             "invalid_confidence")

    geometry_ready = np.ones(count, dtype=bool)
    for value, name in ((camera, "camera"), (joint, "joint")):
        if value is None:
            mark(all_frames, f"missing_{name}")
            geometry_ready[:] = False
        else:
            finite = np.isfinite(value).all(axis=(1, 2))
            mark(~finite, f"nonfinite_{name}")
            geometry_ready &= finite
    if intrinsic is None:
        mark(all_frames, "missing_intrinsic")
        geometry_ready[:] = False
    elif not np.isfinite(intrinsic).all():
        mark(all_frames, "nonfinite_intrinsic")
        geometry_ready[:] = False

    coordinates = [None] * count
    labels = [None] * count
    for index in np.flatnonzero(geometry_ready):
        index = int(index)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                camera_joint = np.linalg.inv(camera[index]) @ joint[index]
        except np.linalg.LinAlgError:
            reasons[index].append("noninvertible_camera")
            continue
        point = camera_joint[:3, 3]
        if not np.isfinite(point).all():
            reasons[index].append("nonfinite_camera_point")
            continue
        if point[2] <= 0:
            reasons[index].append("nonpositive_depth")
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            homogeneous = intrinsic @ point
            xy = homogeneous[:2] / homogeneous[2]
        if not np.isfinite(homogeneous).all() or homogeneous[2] == 0 or not np.isfinite(xy).all():
            reasons[index].append("invalid_projection")
            continue
        x, y = map(float, xy)
        if not (0 <= x < width and 0 <= y < height):
            reasons[index].append("out_of_frame")
            continue
        if reasons[index]:
            continue
        # Comparisons avoid rounding and int overflow at an internal boundary.
        column = 0 if x < width / 3.0 else (1 if x < 2.0 * width / 3.0 else 2)
        row = 0 if y < height / 3.0 else (1 if y < 2.0 * height / 3.0 else 2)
        coordinates[index] = [x, y]
        labels[index] = GRID_NAMES[row][column]

    return {
        "answer": labels,
        "answer_traj": coordinates,
        "reference_valid": [not item for item in reasons],
        "reference_invalid_reason": ["|".join(item) for item in reasons],
    }
