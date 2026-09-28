"""
Digital Companion for Field Drug Testing
Single-file FastAPI backend for Flutter integration.

Architecture:

Flutter
   |
   | multipart/form-data
   | officer_id + GPS + image
   v
FastAPI /analyze
   |
   v
OpenCV preprocessing
   |
   +--> Reference card detection
   +--> Reference colour extraction
   +--> Colour calibration
   +--> Test cassette detection
   +--> Test-area extraction
   +--> Image quality
   +--> LAB colour classification
   +--> Confidence
   +--> SHA-256
   |
   v
JSON + annotated image (base64)
   |
   v
Flutter

IMPORTANT:
The classification prototype colours below are demonstration values only.
They are NOT official drug-specific NCB/NCB-approved colour values.

For actual deployment, replace the prototype colours with experimentally
validated values for the exact field-test kit being used.
"""

import base64
import hashlib
import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ------------------------------------------------------------------------
    # Reference card:
    #
    # WHITE | GRAY | BLACK
    # RED   | GREEN | BLUE
    #
    # RGB values.
    # ------------------------------------------------------------------------
    "reference_colors_rgb": {
        "WHITE": [255, 255, 255],
        "GRAY": [128, 128, 128],
        "BLACK": [0, 0, 0],
        "RED": [255, 0, 0],
        "GREEN": [0, 255, 0],
        "BLUE": [0, 0, 255],
    },

    # ------------------------------------------------------------------------
    # DEMONSTRATION classification prototypes.
    #
    # These are NOT official drug-test colours.
    #
    # Replace with experimentally validated colours from the actual kit.
    # Multiple prototype colours can be supplied for each category.
    # ------------------------------------------------------------------------
    "positive_prototypes_rgb": [
        [180, 70, 60],
        [150, 60, 50],
    ],

    "negative_prototypes_rgb": [
        [215, 215, 215],
        [190, 190, 190],
    ],

    "inconclusive_prototypes_rgb": [
        [128, 128, 128],
        [150, 130, 110],
    ],

    # ------------------------------------------------------------------------
    # Rectangle detection.
    # ------------------------------------------------------------------------
    "min_card_area_ratio": 0.015,
    "max_card_area_ratio": 0.70,

    "min_cassette_area_ratio": 0.005,
    "max_cassette_area_ratio": 0.70,

    "min_rectangle_aspect": 0.30,
    "max_rectangle_aspect": 4.50,

    # ------------------------------------------------------------------------
    # Image processing.
    # ------------------------------------------------------------------------
    "max_image_dimension": 1800,

    # ------------------------------------------------------------------------
    # Image quality.
    # ------------------------------------------------------------------------
    "minimum_quality_score": 55.0,
    "minimum_sharpness": 35.0,

    "minimum_brightness": 35.0,
    "maximum_brightness": 225.0,

    "minimum_contrast": 20.0,

    "maximum_extreme_pixel_ratio": 0.35,

    # ------------------------------------------------------------------------
    # Calibration.
    # ------------------------------------------------------------------------
    "maximum_calibration_error_rgb": 75.0,
    "minimum_calibration_valid_patches": 4,

    # ------------------------------------------------------------------------
    # Classification.
    # ------------------------------------------------------------------------
    "maximum_classification_distance": 70.0,
    "maximum_inconclusive_distance": 90.0,

    "minimum_positive_negative_margin": 8.0,
    "minimum_relative_margin": 0.08,

    "inconclusive_prototype_margin": 8.0,

    "maximum_confidence": 97.0,
    "minimum_confidence_for_binary_result": 52.0,

    # ------------------------------------------------------------------------
    # Test area inside cassette.
    # ------------------------------------------------------------------------
    "test_area_x1": 0.20,
    "test_area_y1": 0.25,
    "test_area_x2": 0.80,
    "test_area_y2": 0.75,

    # ------------------------------------------------------------------------
    # Reference patch extraction.
    # ------------------------------------------------------------------------
    "reference_patch_inner_ratio": 0.55,

    # ------------------------------------------------------------------------
    # Fallback detection.
    # ------------------------------------------------------------------------
    "allow_controlled_fallback_detection": True,

    # ------------------------------------------------------------------------
    # Uploaded image size safety limit.
    # ------------------------------------------------------------------------
    "maximum_upload_bytes": 15 * 1024 * 1024,
}


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Digital Companion for Field Drug Testing",
    description=(
        "OpenCV-based presumptive field-test image analysis API."
    ),
    version="1.0.0",
)

# Prototype CORS configuration.
# IMPORTANT: Restrict allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value)
    )


def safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def rgb_to_hex(rgb):
    values = [
        int(clamp(float(x), 0, 255))
        for x in rgb
    ]

    return "#{:02X}{:02X}{:02X}".format(
        values[0],
        values[1],
        values[2]
    )


def robust_median_rgb(rgb_pixels):
    if rgb_pixels is None:
        return None

    pixels = np.asarray(rgb_pixels)

    if pixels.size == 0:
        return None

    if pixels.ndim == 1:
        if pixels.size != 3:
            return None

        pixels = pixels.reshape(
            1,
            3
        )

    pixels = pixels.reshape(
        -1,
        3
    ).astype(
        np.float32
    )

    pixels = pixels[
        np.isfinite(pixels).all(axis=1)
    ]

    if len(pixels) == 0:
        return None

    brightness = np.mean(
        pixels,
        axis=1
    )

    if len(pixels) >= 20:
        low = np.percentile(
            brightness,
            5
        )

        high = np.percentile(
            brightness,
            95
        )

        filtered = pixels[
            (brightness >= low) &
            (brightness <= high)
        ]

        if len(filtered) >= 5:
            pixels = filtered

    median = np.median(
        pixels,
        axis=0
    )

    return [
        float(x)
        for x in median
    ]


def image_to_bgr(image_rgb):
    return cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )


def bgr_to_rgb(image_bgr):
    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


def order_points(points):
    points = np.asarray(
        points,
        dtype=np.float32
    )

    rect = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    total = points.sum(
        axis=1
    )

    difference = np.diff(
        points,
        axis=1
    ).reshape(-1)

    rect[0] = points[
        np.argmin(total)
    ]

    rect[2] = points[
        np.argmax(total)
    ]

    rect[1] = points[
        np.argmin(difference)
    ]

    rect[3] = points[
        np.argmax(difference)
    ]

    return rect


def rectangle_from_points(points):
    points = np.asarray(
        points,
        dtype=np.float32
    )

    return (
        int(np.min(points[:, 0])),
        int(np.min(points[:, 1])),
        int(np.max(points[:, 0])),
        int(np.max(points[:, 1])),
    )


def rect_area(rect):
    x1, y1, x2, y2 = rect

    return max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )


def rect_center(rect):
    x1, y1, x2, y2 = rect

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def clip_rect(
    rect,
    width,
    height
):
    x1, y1, x2, y2 = rect

    x1 = int(
        clamp(
            x1,
            0,
            width - 1
        )
    )

    y1 = int(
        clamp(
            y1,
            0,
            height - 1
        )
    )

    x2 = int(
        clamp(
            x2,
            x1 + 1,
            width
        )
    )

    y2 = int(
        clamp(
            y2,
            y1 + 1,
            height
        )
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


def rectangles_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    if ix2 <= ix1 or iy2 <= iy1:
        return False

    intersection = (
        (ix2 - ix1) *
        (iy2 - iy1)
    )

    union = (
        rect_area(a) +
        rect_area(b) -
        intersection
    )

    if union <= 0:
        return False

    return (
        intersection / union
    ) > 0.40


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def decode_original_image(original_bytes):
    if not original_bytes:
        raise ValueError(
            "The uploaded image is empty."
        )

    if len(original_bytes) > CONFIG[
        "maximum_upload_bytes"
    ]:
        raise ValueError(
            "The uploaded image is too large."
        )

    image_array = np.frombuffer(
        original_bytes,
        dtype=np.uint8
    )

    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise ValueError(
            "The uploaded file could not be decoded as an image."
        )

    image_rgb = bgr_to_rgb(
        image_bgr
    )

    if (
        image_rgb.ndim != 3 or
        image_rgb.shape[2] != 3
    ):
        raise ValueError(
            "The decoded image is not a valid RGB image."
        )

    return image_rgb


def resize_if_needed(image_rgb):
    height, width = image_rgb.shape[:2]

    maximum = CONFIG[
        "max_image_dimension"
    ]

    if max(
        height,
        width
    ) <= maximum:
        return image_rgb.copy()

    scale = (
        maximum /
        float(max(height, width))
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    return cv2.resize(
        image_rgb,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


def preprocess_image(image_rgb):
    resized = resize_if_needed(
        image_rgb
    )

    bgr = image_to_bgr(
        resized
    )

    denoised_bgr = cv2.bilateralFilter(
        bgr,
        5,
        35,
        35
    )

    processed_rgb = bgr_to_rgb(
        denoised_bgr
    )

    hsv = cv2.cvtColor(
        processed_rgb,
        cv2.COLOR_RGB2HSV
    )

    lab = cv2.cvtColor(
        processed_rgb,
        cv2.COLOR_RGB2LAB
    )

    return {
        "rgb": processed_rgb,
        "bgr": denoised_bgr,
        "hsv": hsv,
        "lab": lab,
    }


# ============================================================================
# IMAGE QUALITY
# ============================================================================

def calculate_image_quality(image_rgb):
    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    extreme_low_ratio = float(
        np.mean(gray <= 12)
    )

    extreme_high_ratio = float(
        np.mean(gray >= 245)
    )

    extreme_ratio = (
        extreme_low_ratio +
        extreme_high_ratio
    )

    sharpness_score = clamp(
        sharpness / 180.0 * 100.0,
        0,
        100
    )

    brightness_score = 100.0 - (
        abs(brightness - 128.0) /
        128.0 *
        100.0
    )

    brightness_score = clamp(
        brightness_score,
        0,
        100
    )

    contrast_score = clamp(
        contrast / 65.0 * 100.0,
        0,
        100
    )

    extreme_score = clamp(
        (
            1.0 -
            extreme_ratio /
            CONFIG["maximum_extreme_pixel_ratio"]
        ) * 100.0,
        0,
        100
    )

    quality_score = (
        0.35 * sharpness_score +
        0.25 * brightness_score +
        0.20 * contrast_score +
        0.20 * extreme_score
    )

    quality_score = clamp(
        quality_score,
        0,
        100
    )

    brightness_ok = (
        CONFIG["minimum_brightness"]
        <= brightness
        <= CONFIG["maximum_brightness"]
    )

    sharpness_ok = (
        sharpness >=
        CONFIG["minimum_sharpness"]
    )

    exposure_ok = (
        extreme_ratio <=
        CONFIG["maximum_extreme_pixel_ratio"]
    )

    contrast_ok = (
        contrast >=
        CONFIG["minimum_contrast"]
    )

    acceptable = (
        quality_score >=
        CONFIG["minimum_quality_score"]
        and
        brightness_ok
        and
        sharpness_ok
        and
        exposure_ok
        and
        contrast_ok
    )

    issues = []

    if not sharpness_ok:
        issues.append(
            "image may be blurry"
        )

    if brightness < CONFIG[
        "minimum_brightness"
    ]:
        issues.append(
            "image is too dark"
        )

    if brightness > CONFIG[
        "maximum_brightness"
    ]:
        issues.append(
            "image is too bright"
        )

    if not contrast_ok:
        issues.append(
            "image contrast is low"
        )

    if not exposure_ok:
        issues.append(
            "too many clipped dark/bright pixels"
        )

    return {
        "score": round(
            float(quality_score),
            2
        ),
        "quality_score": round(
            float(quality_score),
            2
        ),
        "acceptable": bool(
            acceptable
        ),
        "sharpness": round(
            float(sharpness),
            2
        ),
        "brightness": round(
            float(brightness),
            2
        ),
        "contrast": round(
            float(contrast),
            2
        ),
        "extreme_pixel_ratio": round(
            float(extreme_ratio),
            4
        ),
        "brightness_ok": bool(
            brightness_ok
        ),
        "sharpness_ok": bool(
            sharpness_ok
        ),
        "exposure_ok": bool(
            exposure_ok
        ),
        "contrast_ok": bool(
            contrast_ok
        ),
        "issues": issues,
    }


# ============================================================================
# RECTANGLE DETECTION
# ============================================================================

def four_point_rect(contour):
    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter <= 0:
        return None

    approximation = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(approximation) != 4:
        return None

    points = approximation.reshape(
        4,
        2
    ).astype(
        np.float32
    )

    return order_points(
        points
    )


def detect_rectangle_candidates(image_rgb):
    height, width = image_rgb.shape[:2]

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    image_area = float(
        width * height
    )

    for contour in contours:

        area = float(
            cv2.contourArea(contour)
        )

        if area <= 0:
            continue

        area_ratio = (
            area /
            image_area
        )

        if (
            area_ratio < 0.003
            or
            area_ratio > 0.80
        ):
            continue

        points = four_point_rect(
            contour
        )

        if points is None:
            x, y, rw, rh = cv2.boundingRect(
                contour
            )

            if rw <= 0 or rh <= 0:
                continue

            points = np.array(
                [
                    [x, y],
                    [x + rw, y],
                    [x + rw, y + rh],
                    [x, y + rh],
                ],
                dtype=np.float32
            )

        x1, y1, x2, y2 = rectangle_from_points(
            points
        )

        rw = x2 - x1
        rh = y2 - y1

        if rw <= 10 or rh <= 10:
            continue

        aspect = (
            rw /
            float(rh)
        )

        if not (
            CONFIG["min_rectangle_aspect"]
            <= aspect
            <= CONFIG["max_rectangle_aspect"]
        ):
            continue

        rectangular_area = float(
            rw * rh
        )

        fill_ratio = (
            area /
            max(
                rectangular_area,
                1.0
            )
        )

        if fill_ratio < 0.35:
            continue

        candidates.append(
            {
                "rect": clip_rect(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    width,
                    height
                ),
                "area": area,
                "area_ratio": area_ratio,
                "aspect": aspect,
                "fill_ratio": fill_ratio,
                "center": rect_center(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                ),
            }
        )

    return candidates


def detect_reference_card(
    image_rgb,
    candidates
):
    height, width = image_rgb.shape[:2]

    left_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate["center"][0]
            < width * 0.60
        )
        and
        (
            CONFIG["min_card_area_ratio"]
            <= candidate["area_ratio"]
            <= CONFIG["max_card_area_ratio"]
        )
    ]

    if left_candidates:

        left_candidates.sort(
            key=lambda candidate: (
                -candidate["area"],
                candidate["center"][0]
            )
        )

        return (
            left_candidates[0]["rect"],
            "automatic"
        )

    if CONFIG[
        "allow_controlled_fallback_detection"
    ]:

        x1 = int(
            width * 0.03
        )

        x2 = int(
            width * 0.43
        )

        y1 = int(
            height * 0.15
        )

        y2 = int(
            height * 0.85
        )

        return (
            clip_rect(
                (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                width,
                height
            ),
            "controlled-fallback"
        )

    return (
        None,
        "failed"
    )


def detect_test_cassette(
    image_rgb,
    candidates,
    reference_rect
):
    height, width = image_rgb.shape[:2]

    reference_center = rect_center(
        reference_rect
    )

    right_candidates = []

    for candidate in candidates:

        center_x, center_y = candidate[
            "center"
        ]

        if abs(
            center_x -
            reference_center[0]
        ) < 30:
            continue

        if center_x <= width * 0.40:
            continue

        if not (
            CONFIG["min_cassette_area_ratio"]
            <= candidate["area_ratio"]
            <= CONFIG["max_cassette_area_ratio"]
        ):
            continue

        if rectangles_overlap(
            reference_rect,
            candidate["rect"]
        ):
            continue

        right_candidates.append(
            candidate
        )

    if right_candidates:

        right_candidates.sort(
            key=lambda candidate: (
                -candidate["area"],
                -candidate["center"][0]
            )
        )

        return (
            right_candidates[0]["rect"],
            "automatic"
        )

    if CONFIG[
        "allow_controlled_fallback_detection"
    ]:

        x1 = int(
            width * 0.48
        )

        x2 = int(
            width * 0.96
        )

        y1 = int(
            height * 0.20
        )

        y2 = int(
            height * 0.80
        )

        fallback = clip_rect(
            (
                x1,
                y1,
                x2,
                y2
            ),
            width,
            height
        )

        if not rectangles_overlap(
            reference_rect,
            fallback
        ):
            return (
                fallback,
                "controlled-fallback"
            )

    return (
        None,
        "failed"
    )


# ============================================================================
# REFERENCE COLOUR PATCH EXTRACTION
# ============================================================================

def extract_reference_patches(
    image_rgb,
    card_rect
):
    x1, y1, x2, y2 = card_rect

    card = image_rgb[
        y1:y2,
        x1:x2
    ]

    if card.size == 0:
        raise ValueError(
            "Reference card region is empty."
        )

    height, width = card.shape[:2]

    if height < 30 or width < 30:
        raise ValueError(
            "Reference card is too small."
        )

    names = [
        [
            "WHITE",
            "GRAY",
            "BLACK"
        ],
        [
            "RED",
            "GREEN",
            "BLUE"
        ]
    ]

    measured = {}
    patch_rectangles = {}

    inner_ratio = CONFIG[
        "reference_patch_inner_ratio"
    ]

    for row in range(2):

        for column in range(3):

            cell_x1 = int(
                column *
                width /
                3.0
            )

            cell_x2 = int(
                (column + 1) *
                width /
                3.0
            )

            cell_y1 = int(
                row *
                height /
                2.0
            )

            cell_y2 = int(
                (row + 1) *
                height /
                2.0
            )

            cell_width = (
                cell_x2 -
                cell_x1
            )

            cell_height = (
                cell_y2 -
                cell_y1
            )

            crop_width = int(
                cell_width *
                inner_ratio
            )

            crop_height = int(
                cell_height *
                inner_ratio
            )

            center_x = (
                cell_x1 +
                cell_x2
            ) // 2

            center_y = (
                cell_y1 +
                cell_y2
            ) // 2

            patch_x1 = max(
                cell_x1,
                center_x -
                crop_width // 2
            )

            patch_x2 = min(
                cell_x2,
                center_x +
                crop_width // 2
            )

            patch_y1 = max(
                cell_y1,
                center_y -
                crop_height // 2
            )

            patch_y2 = min(
                cell_y2,
                center_y +
                crop_height // 2
            )

            patch = card[
                patch_y1:patch_y2,
                patch_x1:patch_x2
            ]

            rgb = robust_median_rgb(
                patch.reshape(
                    -1,
                    3
                )
            )

            name = names[
                row
            ][
                column
            ]

            if rgb is not None:
                measured[name] = rgb

            patch_rectangles[name] = (
                x1 + patch_x1,
                y1 + patch_y1,
                x1 + patch_x2,
                y1 + patch_y2,
            )

    return (
        measured,
        patch_rectangles
    )


# ============================================================================
# COLOUR CALIBRATION
# ============================================================================

def build_color_calibration(
    measured_colors
):
    expected_colors = CONFIG[
        "reference_colors_rgb"
    ]

    source = []
    target = []

    patch_errors_before = {}

    for name, expected_rgb in expected_colors.items():

        if name not in measured_colors:
            continue

        measured = np.asarray(
            measured_colors[name],
            dtype=np.float64
        )

        expected = np.asarray(
            expected_rgb,
            dtype=np.float64
        )

        source.append(
            measured
        )

        target.append(
            expected
        )

        patch_errors_before[name] = float(
            np.linalg.norm(
                measured -
                expected
            )
        )

    minimum_patches = CONFIG[
        "minimum_calibration_valid_patches"
    ]

    if len(source) < minimum_patches:

        return {
            "valid": False,
            "reason": (
                "Not enough reference colour patches "
                "were successfully detected."
            ),
            "matrix": None,
            "mean_error_before": None,
            "mean_error_after": None,
            "patch_errors_before": patch_errors_before,
            "patch_errors_after": {},
            "valid_patch_count": len(source),
        }

    X = np.asarray(
        source,
        dtype=np.float64
    )

    Y = np.asarray(
        target,
        dtype=np.float64
    )

    # Affine colour transformation:
    #
    # [R G B 1] * M = [R' G' B']
    #
    # This makes calibration mathematically affect the measured test colour.
    X_augmented = np.hstack(
        [
            X,
            np.ones(
                (
                    X.shape[0],
                    1
                ),
                dtype=np.float64
            ),
        ]
    )

    try:

        matrix, residuals, rank, singular_values = (
            np.linalg.lstsq(
                X_augmented,
                Y,
                rcond=None
            )
        )

    except Exception as exc:

        return {
            "valid": False,
            "reason": (
                f"Calibration failed: {exc}"
            ),
            "matrix": None,
            "mean_error_before": None,
            "mean_error_after": None,
            "patch_errors_before": patch_errors_before,
            "patch_errors_after": {},
            "valid_patch_count": len(source),
        }

    predicted = (
        X_augmented @ matrix
    )

    errors_after = np.linalg.norm(
        predicted -
        Y,
        axis=1
    )

    patch_errors_after = {}

    names = [
        name
        for name in expected_colors.keys()
        if name in measured_colors
    ]

    for index, name in enumerate(
        names
    ):
        patch_errors_after[name] = float(
            errors_after[index]
        )

    mean_before = float(
        np.mean(
            list(
                patch_errors_before.values()
            )
        )
    )

    mean_after = float(
        np.mean(
            errors_after
        )
    )

    valid = (
        mean_after <=
        CONFIG[
            "maximum_calibration_error_rgb"
        ]
    )

    return {
        "valid": bool(valid),
        "reason": (
            "Calibration successful."
            if valid
            else
            "Calibration error is above the configured limit."
        ),
        "matrix": matrix,
        "mean_error_before": mean_before,
        "mean_error_after": mean_after,
        "patch_errors_before": patch_errors_before,
        "patch_errors_after": patch_errors_after,
        "valid_patch_count": len(source),
        "rank": int(rank),
    }


def apply_color_calibration(
    rgb_pixels,
    calibration
):
    pixels = np.asarray(
        rgb_pixels,
        dtype=np.float64
    )

    original_shape = pixels.shape

    if pixels.size == 0:
        return pixels

    pixels = pixels.reshape(
        -1,
        3
    )

    matrix = calibration.get(
        "matrix"
    )

    if matrix is None:
        return pixels.reshape(
            original_shape
        )

    augmented = np.hstack(
        [
            pixels,
            np.ones(
                (
                    pixels.shape[0],
                    1
                ),
                dtype=np.float64
            ),
        ]
    )

    corrected = (
        augmented @ matrix
    )

    corrected = np.clip(
        corrected,
        0,
        255
    )

    return corrected.reshape(
        original_shape
    ).astype(
        np.uint8
    )


def calibrate_entire_image(
    image_rgb,
    calibration
):
    if not calibration.get(
        "valid",
        False
    ):
        return image_rgb.copy()

    height, width = image_rgb.shape[:2]

    corrected = apply_color_calibration(
        image_rgb.reshape(
            -1,
            3
        ),
        calibration
    )

    return corrected.reshape(
        height,
        width,
        3
    )


# ============================================================================
# TEST AREA
# ============================================================================

def extract_test_area(
    image_rgb,
    cassette_rect
):
    x1, y1, x2, y2 = cassette_rect

    cassette = image_rgb[
        y1:y2,
        x1:x2
    ]

    if cassette.size == 0:
        return (
            None,
            None
        )

    height, width = cassette.shape[:2]

    if height < 20 or width < 20:
        return (
            None,
            None
        )

    rx1 = int(
        width *
        CONFIG["test_area_x1"]
    )

    ry1 = int(
        height *
        CONFIG["test_area_y1"]
    )

    rx2 = int(
        width *
        CONFIG["test_area_x2"]
    )

    ry2 = int(
        height *
        CONFIG["test_area_y2"]
    )

    rx1 = int(
        clamp(
            rx1,
            0,
            width - 1
        )
    )

    ry1 = int(
        clamp(
            ry1,
            0,
            height - 1
        )
    )

    rx2 = int(
        clamp(
            rx2,
            rx1 + 1,
            width
        )
    )

    ry2 = int(
        clamp(
            ry2,
            ry1 + 1,
            height
        )
    )

    test_area = cassette[
        ry1:ry2,
        rx1:rx2
    ]

    absolute_rect = (
        x1 + rx1,
        y1 + ry1,
        x1 + rx2,
        y1 + ry2
    )

    if test_area.size == 0:
        return (
            None,
            None
        )

    return (
        test_area,
        absolute_rect
    )


def extract_test_colour(
    test_area_rgb
):
    if test_area_rgb is None:
        return None

    pixels = test_area_rgb.reshape(
        -1,
        3
    ).astype(
        np.float32
    )

    if len(pixels) < 10:
        return None

    hsv = cv2.cvtColor(
        test_area_rgb,
        cv2.COLOR_RGB2HSV
    ).reshape(
        -1,
        3
    )

    value = hsv[
        :,
        2
    ].astype(
        np.float32
    )

    low = np.percentile(
        value,
        5
    )

    high = np.percentile(
        value,
        95
    )

    valid = (
        (value >= low)
        &
        (value <= high)
    )

    filtered = pixels[
        valid
    ]

    if len(filtered) < 10:
        filtered = pixels

    return robust_median_rgb(
        filtered
    )


# ============================================================================
# LAB DISTANCE
# ============================================================================

def rgb_to_lab_single(
    rgb
):
    array = np.asarray(
        rgb,
        dtype=np.float32
    ).reshape(
        1,
        1,
        3
    )

    array = np.clip(
        array,
        0,
        255
    ).astype(
        np.uint8
    )

    lab = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2LAB
    )

    return lab[
        0,
        0
    ].astype(
        np.float32
    )


def rgb_to_lab_batch(
    rgb_list
):
    array = np.asarray(
        rgb_list,
        dtype=np.float32
    )

    if array.ndim == 1:
        array = array.reshape(
            1,
            3
        )

    array = np.clip(
        array,
        0,
        255
    ).astype(
        np.uint8
    )

    array = array.reshape(
        -1,
        1,
        3
    )

    lab = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2LAB
    )

    return lab.reshape(
        -1,
        3
    ).astype(
        np.float32
    )


def calculate_category_distance(
    test_rgb,
    prototype_list
):
    if not prototype_list:
        return float("inf")

    test_lab = rgb_to_lab_single(
        test_rgb
    )

    prototypes_lab = rgb_to_lab_batch(
        prototype_list
    )

    distances = np.linalg.norm(
        prototypes_lab -
        test_lab,
        axis=1
    )

    return float(
        np.min(distances)
    )


# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_test_colour(
    test_rgb,
    quality,
    calibration
):
    if test_rgb is None:
        return {
            "result": "INCONCLUSIVE",
            "confidence_percent": 0.0,
            "distances": {},
            "reason": (
                "Test colour could not be extracted. "
                "Retake Image."
            ),
        }

    if not quality[
        "acceptable"
    ]:
        return {
            "result": "INCONCLUSIVE",
            "confidence_percent": round(
                min(
                    quality[
                        "quality_score"
                    ],
                    49.0
                ),
                2
            ),
            "distances": {},
            "reason": (
                "Image quality is insufficient for reliable "
                "colour classification. Retake Image."
            ),
        }

    if not calibration.get(
        "valid",
        False
    ):
        return {
            "result": "INCONCLUSIVE",
            "confidence_percent": 20.0,
            "distances": {},
            "reason": (
                "Reference-card calibration failed or calibration "
                "error is too high. Retake Image."
            ),
        }

    positive_distance = (
        calculate_category_distance(
            test_rgb,
            CONFIG[
                "positive_prototypes_rgb"
            ]
        )
    )

    negative_distance = (
        calculate_category_distance(
            test_rgb,
            CONFIG[
                "negative_prototypes_rgb"
            ]
        )
    )

    inconclusive_distance = (
        calculate_category_distance(
            test_rgb,
            CONFIG[
                "inconclusive_prototypes_rgb"
            ]
        )
    )

    distances = {
        "positive": round(
            positive_distance,
            3
        ),
        "negative": round(
            negative_distance,
            3
        ),
        "inconclusive": round(
            inconclusive_distance,
            3
        ),
    }

    binary = [
        (
            "POSITIVE",
            positive_distance
        ),
        (
            "NEGATIVE",
            negative_distance
        ),
    ]

    binary.sort(
        key=lambda item: item[1]
    )

    best_label = binary[0][0]
    best_distance = binary[0][1]

    second_label = binary[1][0]
    second_distance = binary[1][1]

    absolute_margin = (
        second_distance -
        best_distance
    )

    relative_margin = (
        absolute_margin /
        max(
            second_distance,
            1.0
        )
    )

    if inconclusive_distance <= (
        best_distance +
        CONFIG[
            "inconclusive_prototype_margin"
        ]
    ):
        return {
            "result": "INCONCLUSIVE",
            "confidence_percent": round(
                clamp(
                    40.0 -
                    inconclusive_distance *
                    0.15,
                    10.0,
                    45.0
                ),
                2
            ),
            "distances": distances,
            "reason": (
                "Measured colour is too close to the "
                "configured ambiguous/inconclusive region."
            ),
        }

    if best_distance > CONFIG[
        "maximum_inconclusive_distance"
    ]:
        return {
            "result": "INCONCLUSIVE",
            "confidence_percent": round(
                clamp(
                    35.0 -
                    best_distance *
                    0.05,
                    5.0,
                    35.0
                ),
                2
            ),
            "distances": distances,
            "reason": (
                "Measured colour is too far from the configured "
                "positive and negative prototype colours."
            ),
        }

    if best_distance > CONFIG[
        "maximum_classification_distance"
    ]:
        return {
            "result": "INCONCLUSIVE",
            "confidence_percent": 35.0,
            "distances": distances,
            "reason": (
                "Colour match is outside the reliable "
                "classification distance."
            ),
        }

    if absolute_margin < CONFIG[
        "minimum_positive_negative_margin"
    ]:
        return {
            "result": "INCONCLUSIVE",
            "confidence_percent": round(
                clamp(
                    45.0 +
                    absolute_margin *
                    1.5,
                    20.0,
                    55.0
                ),
                2
            ),
            "distances": distances,
            "reason": (
                f"{best_label} and {second_label} are too close "
                "in LAB colour distance."
            ),
        }

    if relative_margin < CONFIG[
        "minimum_relative_margin"
    ]:
        return {
            "result": "INCONCLUSIVE",
            "confidence_percent": round(
                clamp(
                    45.0 +
                    relative_margin *
                    80.0,
                    20.0,
                    55.0
                ),
                2
            ),
            "distances": distances,
            "reason": (
                "Best-vs-second-best colour margin is too small."
            ),
        }

    distance_score = (
        1.0 -
        best_distance /
        max(
            CONFIG[
                "maximum_classification_distance"
            ],
            1.0
        )
    )

    distance_score = clamp(
        distance_score,
        0.0,
        1.0
    )

    margin_score = clamp(
        relative_margin /
        0.50,
        0.0,
        1.0
    )

    quality_score = (
        quality[
            "quality_score"
        ] /
        100.0
    )

    calibration_error = calibration[
        "mean_error_after"
    ]

    calibration_score = 1.0 - clamp(
        calibration_error /
        max(
            CONFIG[
                "maximum_calibration_error_rgb"
            ],
            1.0
        ),
        0.0,
        1.0
    )

    confidence = (
        0.40 * distance_score +
        0.25 * margin_score +
        0.20 * quality_score +
        0.15 * calibration_score
    )

    confidence *= 100.0

    confidence = clamp(
        confidence,
        0.0,
        CONFIG[
            "maximum_confidence"
        ]
    )

    if confidence < CONFIG[
        "minimum_confidence_for_binary_result"
    ]:
        return {
            "result": "INCONCLUSIVE",
            "confidence_percent": round(
                confidence,
                2
            ),
            "distances": distances,
            "reason": (
                "Colour similarity, margin, image quality, "
                "or calibration quality is insufficient."
            ),
        }

    return {
        "result": best_label,
        "confidence_percent": round(
            confidence,
            2
        ),
        "distances": distances,
        "reason": (
            f"{best_label} has the lowest LAB colour distance "
            f"with an adequate best-vs-second-best margin."
        ),
        "best_category": best_label,
        "second_category": second_label,
        "best_distance": round(
            best_distance,
            3
        ),
        "second_distance": round(
            second_distance,
            3
        ),
        "absolute_margin": round(
            absolute_margin,
            3
        ),
        "relative_margin": round(
            relative_margin,
            4
        ),
    }


# ============================================================================
# ANNOTATED IMAGE
# ============================================================================

def draw_label(
    image,
    text,
    origin,
    color,
    background=(30, 30, 30)
):
    x, y = origin

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.65
    thickness = 2

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            scale,
            thickness
        )
    )

    cv2.rectangle(
        image,
        (
            max(
                0,
                x - 4
            ),
            max(
                0,
                y -
                text_height -
                baseline -
                8
            )
        ),
        (
            min(
                image.shape[1] - 1,
                x +
                text_width +
                8
            ),
            min(
                image.shape[0] - 1,
                y + 5
            )
        ),
        background,
        -1
    )

    cv2.putText(
        image,
        text,
        (
            x,
            y
        ),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def annotate_image(
    image_rgb,
    reference_rect,
    cassette_rect,
    test_rect,
    result,
    confidence
):
    annotated = image_rgb.copy()

    bgr = image_to_bgr(
        annotated
    )

    cyan = (
        255,
        255,
        0
    )

    green = (
        0,
        220,
        0
    )

    red = (
        0,
        0,
        255
    )

    white = (
        255,
        255,
        255
    )

    if reference_rect is not None:

        x1, y1, x2, y2 = reference_rect

        cv2.rectangle(
            bgr,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            cyan,
            2
        )

        draw_label(
            bgr,
            "REFERENCE CARD",
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            white
        )

    if cassette_rect is not None:

        x1, y1, x2, y2 = cassette_rect

        cv2.rectangle(
            bgr,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            green,
            2
        )

        draw_label(
            bgr,
            "TEST CASSETTE",
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            white
        )

    if test_rect is not None:

        x1, y1, x2, y2 = test_rect

        cv2.rectangle(
            bgr,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            red,
            2
        )

        label_y = min(
            bgr.shape[0] - 10,
            y2 + 25
        )

        draw_label(
            bgr,
            result,
            (
                x1,
                label_y
            ),
            white
        )

        draw_label(
            bgr,
            f"Confidence: {confidence:.1f}%",
            (
                x1,
                min(
                    bgr.shape[0] - 10,
                    label_y + 28
                )
            ),
            white
        )

    return bgr_to_rgb(
        bgr
    )


def encode_annotated_image(
    image_rgb
):
    image_bgr = image_to_bgr(
        image_rgb
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            90,
        ]
    )

    if not success:
        raise ValueError(
            "Could not encode annotated image."
        )

    return base64.b64encode(
        encoded.tobytes()
    ).decode(
        "utf-8"
    )


# ============================================================================
# DIGITAL RECORD
# ============================================================================

def build_digital_record(
    test_id,
    result,
    confidence,
    officer_id,
    timestamp,
    latitude,
    longitude,
    accuracy_meters,
    sha256,
    quality,
    reference_rect,
    measured_reference_colors,
    calibration,
    test_rect,
    test_rgb,
    classification
):
    if (
        latitude is not None
        and
        longitude is not None
    ):
        gps = {
            "latitude": latitude,
            "longitude": longitude,
            "accuracy_meters": accuracy_meters,
        }
    else:
        gps = {
            "latitude": None,
            "longitude": None,
            "accuracy_meters": None,
            "status": "GPS unavailable",
        }

    return {
        "test_id": test_id,
        "result": result,
        "confidence_percent": round(
            float(confidence),
            2
        ),
        "officer_id": officer_id,
        "timestamp_utc": timestamp,
        "gps": gps,
        "sha256": sha256,

        "image_quality": quality,

        "reference_card": {
            "detected": reference_rect is not None,
            "rectangle_xyxy": (
                list(reference_rect)
                if reference_rect is not None
                else None
            ),
            "expected_colors_rgb": CONFIG[
                "reference_colors_rgb"
            ],
            "measured_colors_rgb": measured_reference_colors,
            "calibration_error": calibration.get(
                "mean_error_after"
            ),
        },

        "calibration": {
            "valid": bool(
                calibration.get(
                    "valid",
                    False
                )
            ),
            "reason": calibration.get(
                "reason"
            ),
            "valid_patch_count": calibration.get(
                "valid_patch_count",
                0
            ),
            "mean_error_before_rgb": calibration.get(
                "mean_error_before"
            ),
            "mean_error_after_rgb": calibration.get(
                "mean_error_after"
            ),
            "patch_errors_before_rgb": calibration.get(
                "patch_errors_before",
                {}
            ),
            "patch_errors_after_rgb": calibration.get(
                "patch_errors_after",
                {}
            ),
        },

        "test_area": {
            "rectangle_xyxy": (
                list(test_rect)
                if test_rect is not None
                else None
            ),
            "rgb": (
                [
                    round(
                        float(x),
                        2
                    )
                    for x in test_rgb
                ]
                if test_rgb is not None
                else None
            ),
            "hex": (
                rgb_to_hex(
                    test_rgb
                )
                if test_rgb is not None
                else None
            ),
        },

        "classification": {
            "method": (
                "LAB_COLOUR_DISTANCE"
            ),
            "distances": classification.get(
                "distances",
                {}
            ),
            "reason": classification.get(
                "reason"
            ),
            "best_category": classification.get(
                "best_category"
            ),
            "second_category": classification.get(
                "second_category"
            ),
            "best_distance": classification.get(
                "best_distance"
            ),
            "second_distance": classification.get(
                "second_distance"
            ),
            "absolute_margin": classification.get(
                "absolute_margin"
            ),
            "relative_margin": classification.get(
                "relative_margin"
            ),
        },

        "presumptive_result": result in [
            "POSITIVE",
            "NEGATIVE"
        ],

        "requires_lab_confirmation": True,

        "scientific_limitation": (
            "This is a presumptive field-test result and does "
            "not replace laboratory confirmatory testing."
        ),
    }


# ============================================================================
# COMPLETE IMAGE ANALYSIS PIPELINE
# ============================================================================

def process_image(
    original_bytes,
    officer_id,
    latitude,
    longitude,
    accuracy_meters
):
    timestamp = utc_now_iso()

    test_id = (
        "TEST-" +
        uuid.uuid4().hex[:12].upper()
    )

    # ------------------------------------------------------------------------
    # SHA-256 MUST be calculated from original uploaded bytes.
    # ------------------------------------------------------------------------
    sha256 = hashlib.sha256(
        original_bytes
    ).hexdigest()

    # ------------------------------------------------------------------------
    # Decode original image.
    # ------------------------------------------------------------------------
    original_rgb = decode_original_image(
        original_bytes
    )

    # ------------------------------------------------------------------------
    # Create processing copy.
    # The original bytes remain untouched for SHA-256.
    # ------------------------------------------------------------------------
    processed = preprocess_image(
        original_rgb
    )

    image_rgb = processed[
        "rgb"
    ]

    quality = calculate_image_quality(
        image_rgb
    )

    # ------------------------------------------------------------------------
    # Detect major rectangles.
    # ------------------------------------------------------------------------
    candidates = detect_rectangle_candidates(
        image_rgb
    )

    reference_rect, reference_method = (
        detect_reference_card(
            image_rgb,
            candidates
        )
    )

    if reference_rect is None:

        classification = {
            "result": "INCONCLUSIVE",
            "confidence_percent": 10.0,
            "distances": {},
            "reason": (
                "Reference card could not be detected. "
                "Retake Image."
            ),
        }

        annotated = image_rgb.copy()

        encoded_image = encode_annotated_image(
            annotated
        )

        record = build_digital_record(
            test_id,
            "INCONCLUSIVE",
            10.0,
            officer_id,
            timestamp,
            latitude,
            longitude,
            accuracy_meters,
            sha256,
            quality,
            None,
            {},
            {
                "valid": False,
                "reason": "Reference card missing.",
                "valid_patch_count": 0,
                "mean_error_after": None,
            },
            None,
            None,
            classification
        )

        return {
            "success": True,
            "test_id": test_id,
            "result": "INCONCLUSIVE",
            "confidence_percent": 10.0,
            "officer_id": officer_id,
            "timestamp_utc": timestamp,
            "gps": record["gps"],
            "sha256": sha256,
            "image_quality": quality,
            "reference_card": {
                "detected": False,
                "detection_method": reference_method,
            },
            "classification": classification,
            "presumptive_result": False,
            "requires_lab_confirmation": True,
            "annotated_image_base64": encoded_image,
            "digital_record": record,
            "message": "Reference card missing. Retake Image.",
        }

    # ------------------------------------------------------------------------
    # Extract six reference colours.
    # ------------------------------------------------------------------------
    try:
        (
            measured_reference_colors,
            patch_rectangles
        ) = extract_reference_patches(
            image_rgb,
            reference_rect
        )

    except Exception as exc:

        measured_reference_colors = {}

        patch_rectangles = {}

    # ------------------------------------------------------------------------
    # Build affine colour calibration.
    # ------------------------------------------------------------------------
    calibration = build_color_calibration(
        measured_reference_colors
    )

    # ------------------------------------------------------------------------
    # Detect cassette.
    # ------------------------------------------------------------------------
    cassette_rect, cassette_method = (
        detect_test_cassette(
            image_rgb,
            candidates,
            reference_rect
        )
    )

    if cassette_rect is None:

        classification = {
            "result": "INCONCLUSIVE",
            "confidence_percent": 10.0,
            "distances": {},
            "reason": (
                "Test cassette could not be detected. "
                "Retake Image."
            ),
        }

        annotated = annotate_image(
            image_rgb,
            reference_rect,
            None,
            None,
            "INCONCLUSIVE",
            10.0
        )

        encoded_image = encode_annotated_image(
            annotated
        )

        record = build_digital_record(
            test_id,
            "INCONCLUSIVE",
            10.0,
            officer_id,
            timestamp,
            latitude,
            longitude,
            accuracy_meters,
            sha256,
            quality,
            reference_rect,
            measured_reference_colors,
            calibration,
            None,
            None,
            classification
        )

        return {
            "success": True,
            "test_id": test_id,
            "result": "INCONCLUSIVE",
            "confidence_percent": 10.0,
            "officer_id": officer_id,
            "timestamp_utc": timestamp,
            "gps": record["gps"],
            "sha256": sha256,
            "image_quality": quality,
            "reference_card": {
                "detected": True,
                "detection_method": reference_method,
                "calibration_error": calibration.get(
                    "mean_error_after"
                ),
            },
            "test_area": None,
            "classification": classification,
            "presumptive_result": False,
            "requires_lab_confirmation": True,
            "annotated_image_base64": encoded_image,
            "digital_record": record,
            "message": "Test cassette missing. Retake Image.",
        }

    # ------------------------------------------------------------------------
    # Calibration must affect the image used for test colour extraction.
    # ------------------------------------------------------------------------
    calibrated_image = calibrate_entire_image(
        image_rgb,
        calibration
    )

    # ------------------------------------------------------------------------
    # Extract test area.
    # ------------------------------------------------------------------------
    test_area, test_rect = extract_test_area(
        calibrated_image,
        cassette_rect
    )

    if test_area is None:

        classification = {
            "result": "INCONCLUSIVE",
            "confidence_percent": 10.0,
            "distances": {},
            "reason": (
                "Test area could not be reliably extracted. "
                "Retake Image."
            ),
        }

        annotated = annotate_image(
            calibrated_image,
            reference_rect,
            cassette_rect,
            None,
            "INCONCLUSIVE",
            10.0
        )

        encoded_image = encode_annotated_image(
            annotated
        )

        record = build_digital_record(
            test_id,
            "INCONCLUSIVE",
            10.0,
            officer_id,
            timestamp,
            latitude,
            longitude,
            accuracy_meters,
            sha256,
            quality,
            reference_rect,
            measured_reference_colors,
            calibration,
            None,
            None,
            classification
        )

        return {
            "success": True,
            "test_id": test_id,
            "result": "INCONCLUSIVE",
            "confidence_percent": 10.0,
            "officer_id": officer_id,
            "timestamp_utc": timestamp,
            "gps": record["gps"],
            "sha256": sha256,
            "image_quality": quality,
            "reference_card": {
                "detected": True,
                "detection_method": reference_method,
                "calibration_error": calibration.get(
                    "mean_error_after"
                ),
            },
            "test_area": None,
            "classification": classification,
            "presumptive_result": False,
            "requires_lab_confirmation": True,
            "annotated_image_base64": encoded_image,
            "digital_record": record,
            "message": "Test area missing. Retake Image.",
        }

    # ------------------------------------------------------------------------
    # Robust calibrated test colour.
    # ------------------------------------------------------------------------
    test_rgb = extract_test_colour(
        test_area
    )

    # ------------------------------------------------------------------------
    # Classification.
    # ------------------------------------------------------------------------
    classification = classify_test_colour(
        test_rgb,
        quality,
        calibration
    )

    result = classification[
        "result"
    ]

    confidence = classification[
        "confidence_percent"
    ]

    # ------------------------------------------------------------------------
    # Annotation.
    # ------------------------------------------------------------------------
    annotated = annotate_image(
        calibrated_image,
        reference_rect,
        cassette_rect,
        test_rect,
        result,
        confidence
    )

    encoded_image = encode_annotated_image(
        annotated
    )

    # ------------------------------------------------------------------------
    # Digital record.
    # ------------------------------------------------------------------------
    record = build_digital_record(
        test_id,
        result,
        confidence,
        officer_id,
        timestamp,
        latitude,
        longitude,
        accuracy_meters,
        sha256,
        quality,
        reference_rect,
        measured_reference_colors,
        calibration,
        test_rect,
        test_rgb,
        classification
    )

    response = {
        "success": True,

        "test_id": test_id,

        "result": result,

        "confidence_percent": round(
            float(confidence),
            2
        ),

        "officer_id": officer_id,

        "timestamp_utc": timestamp,

        "gps": record["gps"],

        "sha256": sha256,

        "image_quality": quality,

        "reference_card": {
            "detected": True,
            "detection_method": reference_method,
            "rectangle_xyxy": list(
                reference_rect
            ),
            "measured_colors_rgb": (
                measured_reference_colors
            ),
            "calibration_error": calibration.get(
                "mean_error_after"
            ),
            "calibration_valid": calibration.get(
                "valid",
                False
            ),
        },

        "test_cassette": {
            "detected": True,
            "detection_method": cassette_method,
            "rectangle_xyxy": list(
                cassette_rect
            ),
        },

        "test_area": {
            "rectangle_xyxy": (
                list(test_rect)
                if test_rect is not None
                else None
            ),
            "rgb": (
                [
                    round(
                        float(x),
                        2
                    )
                    for x in test_rgb
                ]
                if test_rgb is not None
                else None
            ),
            "hex": (
                rgb_to_hex(
                    test_rgb
                )
                if test_rgb is not None
                else None
            ),
        },

        "classification": {
            "method": "LAB_COLOUR_DISTANCE",
            "distances": classification.get(
                "distances",
                {}
            ),
            "reason": classification.get(
                "reason"
            ),
            "best_category": classification.get(
                "best_category"
            ),
            "second_category": classification.get(
                "second_category"
            ),
            "best_distance": classification.get(
                "best_distance"
            ),
            "second_distance": classification.get(
                "second_distance"
            ),
            "absolute_margin": classification.get(
                "absolute_margin"
            ),
            "relative_margin": classification.get(
                "relative_margin"
            ),
        },

        "presumptive_result": result in [
            "POSITIVE",
            "NEGATIVE"
        ],

        "requires_lab_confirmation": True,

        "scientific_limitation": (
            "This is a presumptive field-test result and does "
            "not replace laboratory confirmatory testing."
        ),

        "annotated_image_base64": encoded_image,

        "digital_record": record,
    }

    if result == "INCONCLUSIVE":
        response["message"] = (
            "INCONCLUSIVE. Retake Image."
        )

    else:
        response["message"] = (
            "Analysis completed. "
            "This remains a presumptive field-test result."
        )

    return response


# ============================================================================
# REQUEST VALIDATION
# ============================================================================

def validate_coordinates(
    latitude,
    longitude
):
    if latitude is None:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": "Missing latitude",
                "message": (
                    "Latitude is required."
                ),
            }
        )

    if longitude is None:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": "Missing longitude",
                "message": (
                    "Longitude is required."
                ),
            }
        )

    if not (
        -90.0 <= latitude <= 90.0
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": "Invalid latitude",
                "message": (
                    "Latitude must be between -90 and 90."
                ),
            }
        )

    if not (
        -180.0 <= longitude <= 180.0
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": "Invalid longitude",
                "message": (
                    "Longitude must be between -180 and 180."
                ),
            }
        )


# ============================================================================
# HEALTH ENDPOINT
# ============================================================================

@app.get(
    "/health"
)
async def health():
    return {
        "status": "ok"
    }


# ============================================================================
# ANALYZE ENDPOINT
# ============================================================================

@app.post(
    "/analyze"
)
async def analyze(
    officer_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    accuracy_meters: Optional[float] = Form(None),
    image: UploadFile = File(...)
):
    try:

        # ------------------------------------------------------------
        # Officer ID validation.
        # ------------------------------------------------------------
        if officer_id is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "success": False,
                    "error": "Missing officer_id",
                    "message": (
                        "officer_id is required."
                    ),
                }
            )

        officer_id = officer_id.strip()

        if not officer_id:
            raise HTTPException(
                status_code=422,
                detail={
                    "success": False,
                    "error": "Missing officer_id",
                    "message": (
                        "officer_id cannot be empty."
                    ),
                }
            )

        # ------------------------------------------------------------
        # GPS validation.
        # ------------------------------------------------------------
        latitude_value = safe_float(
            latitude
        )

        longitude_value = safe_float(
            longitude
        )

        accuracy_value = safe_float(
            accuracy_meters
        )

        validate_coordinates(
            latitude_value,
            longitude_value
        )

        # ------------------------------------------------------------
        # Image validation.
        # ------------------------------------------------------------
        if image is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "success": False,
                    "error": "Missing image",
                    "message": (
                        "An image file is required."
                    ),
                }
            )

        original_bytes = await image.read()

        if not original_bytes:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "Invalid image",
                    "message": (
                        "The uploaded image is empty."
                    ),
                }
            )

        # ------------------------------------------------------------
        # Process.
        # ------------------------------------------------------------
        result = process_image(
            original_bytes,
            officer_id,
            latitude_value,
            longitude_value,
            accuracy_value
        )

        return JSONResponse(
            status_code=200,
            content=result
        )

    except HTTPException:
        raise

    except ValueError as exc:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid image",
                "message": str(exc),
            }
        )

    except cv2.error:

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "OpenCV processing failure",
                "message": (
                    "The uploaded image could not be processed "
                    "by the computer-vision pipeline."
                ),
            }
        )

    except Exception:

        # Do not expose traceback/internal implementation details.
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Processing failed",
                "message": (
                    "The image could not be processed. "
                    "Please retake the image and try again."
                ),
            }
        )


# ============================================================================
# OPTIONAL ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "Digital Companion for Field Drug Testing",
        "status": "running",
        "health": "/health",
        "analyze": "POST /analyze",
        "docs": "/docs",
    }


# ============================================================================
# DIRECT EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
