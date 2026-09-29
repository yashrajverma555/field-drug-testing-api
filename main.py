import base64
import datetime
import hashlib
import json
import logging
import math
import secrets
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------------------
# Logging & Server Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("field_drug_testing_api")

app = FastAPI(
    title="Digital Companion for Field Drug Testing API",
    version="2.0.0",
    description="Production-grade colourimetric image analysis backend for field drug test kits."
)

# CORS configuration to fully support Flutter Web and mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Test Kit Color Reference Configuration
# ------------------------------------------------------------------------------
# Conservative calibration placeholder: when None, the system reliably measures
# and compares colors, returning INCONCLUSIVE with clear scientific rationale.
TEST_KIT_CONFIG: Dict[str, Any] = {
    "name": "default",
    "positive_reference_rgb": None,  # e.g., [180, 20, 60]
    "negative_reference_rgb": None,  # e.g., [240, 240, 240]
    "positive_lab_threshold": None,  # e.g., 25.0
    "negative_lab_threshold": None,  # e.g., 25.0
}


# ------------------------------------------------------------------------------
# Utility & Color Science Functions
# ------------------------------------------------------------------------------
def calculate_sha256(data: bytes) -> str:
    """Computes SHA-256 directly on the exact raw binary bytes."""
    return hashlib.sha256(data).hexdigest()


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Converts RGB integers to standard hexadecimal color representation."""
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def rgb_to_lab(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Converts RGB floats to CIE-LAB color space using OpenCV."""
    pixel_bgr = np.uint8([[[int(round(b)), int(round(g)), int(round(r))]]])
    lab_pixel = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)
    return (float(lab_pixel[0, 0, 0]), float(lab_pixel[0, 0, 1]), float(lab_pixel[0, 0, 2]))


def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Converts RGB floats to HSV color space using OpenCV."""
    pixel_bgr = np.uint8([[[int(round(b)), int(round(g)), int(round(r))]]])
    hsv_pixel = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
    return (float(hsv_pixel[0, 0, 0]), float(hsv_pixel[0, 0, 1]), float(hsv_pixel[0, 0, 2]))


# ------------------------------------------------------------------------------
# Image Quality Assessment
# ------------------------------------------------------------------------------
def calculate_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """Assesses blur, exposure, contrast, and resolution without rejecting usable images."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    total_pixels = float(width * height)
    overexposure_percent = float(np.sum(gray >= 250) / total_pixels * 100.0)
    underexposure_percent = float(np.sum(gray <= 10) / total_pixels * 100.0)

    warnings: List[str] = []
    if blur_score < 40.0:
        warnings.append("Image may be slightly blurry")
    if brightness < 35.0:
        warnings.append("Low ambient lighting detected")
    elif brightness > 225.0:
        warnings.append("High ambient brightness / glare detected")
    if overexposure_percent > 15.0:
        warnings.append("Significant overexposed glare areas present")
    if underexposure_percent > 20.0:
        warnings.append("Significant dark shadow areas present")

    score_components = [
        min(100.0, max(0.0, blur_score / 2.0)),
        min(100.0, max(0.0, 100.0 - abs(brightness - 128.0) * 0.7)),
        min(100.0, max(0.0, contrast * 1.5)),
        min(100.0, max(0.0, 100.0 - overexposure_percent * 3.0)),
        min(100.0, max(0.0, 100.0 - underexposure_percent * 3.0)),
    ]
    quality_score = float(round(sum(score_components) / len(score_components), 2))
    acceptable = quality_score >= 30.0 and blur_score >= 20.0

    return {
        "acceptable": acceptable,
        "score": quality_score,
        "resolution": {"width": width, "height": height},
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "overexposure_percent": round(overexposure_percent, 2),
        "underexposure_percent": round(underexposure_percent, 2),
        "warnings": warnings,
    }


# ------------------------------------------------------------------------------
# Robust ROI Color Measurement
# ------------------------------------------------------------------------------
def measure_color(image: np.ndarray, roi_xyxy: List[int], margin_ratio: float = 0.15) -> Dict[str, Any]:
    """Measures median color by shaving borders and filtering glare/dust outliers."""
    x1, y1, x2, y2 = roi_xyxy
    h_img, w_img = image.shape[:2]

    x1 = max(0, min(w_img - 1, x1))
    x2 = max(x1 + 1, min(w_img, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(y1 + 1, min(h_img, y2))

    roi_w = x2 - x1
    roi_h = y2 - y1

    dx = int(roi_w * margin_ratio)
    dy = int(roi_h * margin_ratio)

    cx1 = x1 + dx
    cx2 = max(cx1 + 1, x2 - dx)
    cy1 = y1 + dy
    cy2 = max(cy1 + 1, y2 - dy)

    central_patch = image[cy1:cy2, cx1:cx2]
    if central_patch.size == 0:
        central_patch = image[y1:y2, x1:x2]

    rgb_patch = cv2.cvtColor(central_patch, cv2.COLOR_BGR2RGB)
    pixels = rgb_patch.reshape(-1, 3).astype(np.float32)

    # Filter out luminance outliers (glare, specular highlights, dark edge pixels)
    if len(pixels) > 10:
        luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        p10 = np.percentile(luminance, 10)
        p90 = np.percentile(luminance, 90)
        mask = (luminance >= p10) & (luminance <= p90)
        filtered_pixels = pixels[mask]
        if len(filtered_pixels) < 5:
            filtered_pixels = pixels
    else:
        filtered_pixels = pixels

    median_rgb = [round(float(v), 2) for v in np.median(filtered_pixels, axis=0)]
    mean_rgb = [round(float(v), 2) for v in np.mean(filtered_pixels, axis=0)]

    r, g, b = median_rgb[0], median_rgb[1], median_rgb[2]
    hsv_vals = rgb_to_hsv(r, g, b)
    lab_vals = rgb_to_lab(r, g, b)
    hex_code = rgb_to_hex(int(round(r)), int(round(g)), int(round(b)))

    brightness = round(float(0.299 * r + 0.587 * g + 0.114 * b), 2)
    saturation = round(float(hsv_vals[1]), 2)

    return {
        "rgb": median_rgb,
        "mean_rgb": mean_rgb,
        "hsv": [round(v, 2) for v in hsv_vals],
        "lab": [round(v, 2) for v in lab_vals],
        "hex": hex_code,
        "brightness": brightness,
        "saturation": saturation,
    }


# ------------------------------------------------------------------------------
# Left Reference Swatch Detection
# ------------------------------------------------------------------------------
def detect_reference_card(image: np.ndarray) -> Dict[str, Any]:
    """Detects the large colored reference swatch located on the left side."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect: Optional[List[int]] = None
    best_score = -1.0
    image_area = float(w * h)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.015:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        cx = x + (cw / 2.0)

        # Swatch is located in the left half of the image
        if cx > w * 0.55:
            continue

        aspect_ratio = float(cw) / float(ch) if ch > 0 else 0.0
        if aspect_ratio < 0.25 or aspect_ratio > 3.5:
            continue

        rect_area = float(cw * ch)
        rectangularity = area / rect_area if rect_area > 0 else 0.0
        if rectangularity < 0.45:
            continue

        left_preference = 1.0 - (cx / (w * 0.55))
        score = (area / image_area) * 2.0 + rectangularity * 1.5 + left_preference * 1.0

        if score > best_score:
            best_score = score
            best_rect = [x, y, x + cw, y + ch]

    if best_rect is not None:
        color_data = measure_color(image, best_rect, margin_ratio=0.15)
        return {
            "detected": True,
            "detection_method": "contour_morphological_left",
            "rectangle_xyxy": best_rect,
            "color": color_data,
        }

    # Controlled left-sector ROI fallback if contours are blurred or diffused
    left_fallback = [
        int(w * 0.06),
        int(h * 0.20),
        int(w * 0.44),
        int(h * 0.80)
    ]
    color_data = measure_color(image, left_fallback, margin_ratio=0.15)
    return {
        "detected": True,
        "detection_method": "left_sector_fallback",
        "rectangle_xyxy": left_fallback,
        "color": color_data,
    }


# ------------------------------------------------------------------------------
# Right Test Cassette Detection
# ------------------------------------------------------------------------------
def detect_test_cassette(image: np.ndarray) -> Dict[str, Any]:
    """Detects the rectangular test cassette located on the right side."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect: Optional[List[int]] = None
    best_score = -1.0
    image_area = float(w * h)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.02:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        cx = x + (cw / 2.0)

        # Cassette is located on the right half
        if cx < w * 0.45:
            continue

        aspect_ratio = float(cw) / float(ch) if ch > 0 else 0.0
        if aspect_ratio < 0.2 or aspect_ratio > 3.5:
            continue

        rect_area = float(cw * ch)
        rectangularity = area / rect_area if rect_area > 0 else 0.0
        if rectangularity < 0.40:
            continue

        right_preference = (cx - (w * 0.45)) / (w * 0.55)
        score = (area / image_area) * 2.0 + rectangularity * 1.5 + right_preference * 1.0

        if score > best_score:
            best_score = score
            best_rect = [x, y, x + cw, y + ch]

    if best_rect is not None:
        return {
            "detected": True,
            "detection_method": "contour_canny_right",
            "rectangle_xyxy": best_rect,
        }

    # Controlled right-sector ROI fallback
    right_fallback = [
        int(w * 0.55),
        int(h * 0.20),
        int(w * 0.94),
        int(h * 0.80)
    ]
    return {
        "detected": True,
        "detection_method": "right_sector_fallback",
        "rectangle_xyxy": right_fallback,
    }


# ------------------------------------------------------------------------------
# Reaction / Test Area Detection
# ------------------------------------------------------------------------------
def detect_test_area(image: np.ndarray, cassette_xyxy: List[int]) -> Dict[str, Any]:
    """Finds the central reaction area inside the detected cassette while avoiding borders and text."""
    cx1, cy1, cx2, cy2 = cassette_xyxy
    cw = cx2 - cx1
    ch = cy2 - cy1

    tx1 = int(cx1 + cw * 0.28)
    tx2 = int(cx1 + cw * 0.72)
    ty1 = int(cy1 + ch * 0.32)
    ty2 = int(cy1 + ch * 0.68)

    test_area_xyxy = [tx1, ty1, tx2, ty2]
    color_data = measure_color(image, test_area_xyxy, margin_ratio=0.10)

    return {
        "detected": True,
        "rectangle_xyxy": test_area_xyxy,
        "color": color_data,
    }


# ------------------------------------------------------------------------------
# Color Comparison & Metric Calculation
# ------------------------------------------------------------------------------
def compare_colors(ref_color: Dict[str, Any], test_color: Dict[str, Any]) -> Dict[str, Any]:
    """Performs perceptual color difference calculation in CIE-LAB and RGB spaces."""
    ref_rgb = np.array(ref_color["rgb"], dtype=np.float32)
    test_rgb = np.array(test_color["rgb"], dtype=np.float32)

    ref_lab = np.array(ref_color["lab"], dtype=np.float32)
    test_lab = np.array(test_color["lab"], dtype=np.float32)

    ref_hsv = ref_color["hsv"]
    test_hsv = test_color["hsv"]

    rgb_dist = float(np.linalg.norm(ref_rgb - test_rgb))
    lab_dist = float(np.linalg.norm(ref_lab - test_lab))

    brightness_diff = float(abs(ref_color["brightness"] - test_color["brightness"]))
    saturation_diff = float(abs(ref_color["saturation"] - test_color["saturation"]))

    h1, h2 = float(ref_hsv[0]), float(test_hsv[0])
    raw_hue_diff = abs(h1 - h2)
    hue_diff = min(raw_hue_diff, 180.0 - raw_hue_diff)

    return {
        "reference_rgb": ref_color["rgb"],
        "test_rgb": test_color["rgb"],
        "reference_lab": ref_color["lab"],
        "test_lab": test_color["lab"],
        "rgb_distance": round(rgb_dist, 2),
        "lab_distance": round(lab_dist, 2),
        "brightness_difference": round(brightness_diff, 2),
        "saturation_difference": round(saturation_diff, 2),
        "hue_difference": round(hue_diff, 2),
        "reason": "Relative colourimetric difference between reference swatch and reaction area computed via perceptual CIE-LAB space."
    }


# ------------------------------------------------------------------------------
# Classification Logic
# ------------------------------------------------------------------------------
def classify_result(comparison: Dict[str, Any]) -> Tuple[str, float, bool, str]:
    """Classifies presumptive result using configured thresholds or defaults to INCONCLUSIVE."""
    pos_rgb = TEST_KIT_CONFIG.get("positive_reference_rgb")
    neg_rgb = TEST_KIT_CONFIG.get("negative_reference_rgb")
    pos_thresh = TEST_KIT_CONFIG.get("positive_lab_threshold")
    neg_thresh = TEST_KIT_CONFIG.get("negative_lab_threshold")

    if pos_rgb is None or neg_rgb is None or pos_thresh is None or neg_thresh is None:
        reason = "Test-kit-specific positive/negative color calibration is not configured."
        return ("INCONCLUSIVE", 20.0, False, reason)

    test_rgb = np.array(comparison["test_rgb"], dtype=np.float32)
    pos_lab = np.array(rgb_to_lab(pos_rgb[0], pos_rgb[1], pos_rgb[2]), dtype=np.float32)
    neg_lab = np.array(rgb_to_lab(neg_rgb[0], neg_rgb[1], neg_rgb[2]), dtype=np.float32)
    test_lab = np.array(comparison["test_lab"], dtype=np.float32)

    dist_pos = float(np.linalg.norm(test_lab - pos_lab))
    dist_neg = float(np.linalg.norm(test_lab - neg_lab))

    if dist_pos <= pos_thresh and dist_pos < dist_neg:
        confidence = max(50.0, min(99.0, 100.0 - (dist_pos / pos_thresh) * 45.0))
        return ("POSITIVE", round(confidence, 1), True, "Test colour closely matches calibrated positive reagent reaction profile.")
    elif dist_neg <= neg_thresh and dist_neg < dist_pos:
        confidence = max(50.0, min(99.0, 100.0 - (dist_neg / neg_thresh) * 45.0))
        return ("NEGATIVE", round(confidence, 1), True, "Test colour closely matches calibrated negative reagent reaction profile.")
    else:
        return ("INCONCLUSIVE", 30.0, False, "Reaction colour lies outside validated positive/negative calibration thresholds.")


# ------------------------------------------------------------------------------
# Image Annotation
# ------------------------------------------------------------------------------
def create_annotated_image(
    image: np.ndarray,
    ref_xyxy: List[int],
    ref_color: Dict[str, Any],
    cassette_xyxy: List[int],
    test_xyxy: List[int],
    test_color: Dict[str, Any]
) -> str:
    """Generates an annotated copy with Magenta reference, Green cassette, and Red test area."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # MAGENTA: Reference Swatch
    rx1, ry1, rx2, ry2 = ref_xyxy
    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 0, 255), 3)

    # GREEN: Test Cassette
    cx1, cy1, cx2, cy2 = cassette_xyxy
    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)

    # RED: Test / Reaction Area
    tx1, ty1, tx2, ty2 = test_xyxy
    cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (0, 0, 255), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.9, w / 1200.0))
    thickness = 2

    # Draw Reference Label
    ref_label1 = "REFERENCE"
    ref_label2 = f"RGB: {ref_color['rgb']}"
    ref_label3 = f"HEX: {ref_color['hex']}"
    cv2.putText(annotated, ref_label1, (rx1, max(25, ry1 - 35)), font, font_scale, (255, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(annotated, ref_label2, (rx1, max(45, ry1 - 18)), font, font_scale * 0.85, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, ref_label3, (rx1, max(65, ry1 - 3)), font, font_scale * 0.85, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw Cassette Label
    cv2.putText(annotated, "TEST CASSETTE", (cx1, max(25, cy1 - 10)), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # Draw Test Area Label
    test_label1 = "TEST AREA"
    test_label2 = f"RGB: {test_color['rgb']}"
    test_label3 = f"HEX: {test_color['hex']}"
    cv2.putText(annotated, test_label1, (tx1, max(30, ty1 - 35)), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(annotated, test_label2, (tx1, max(50, ty1 - 18)), font, font_scale * 0.85, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, test_label3, (tx1, max(70, ty1 - 3)), font, font_scale * 0.85, (255, 255, 255), 1, cv2.LINE_AA)

    success, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Lightweight health check endpoint for monitoring."""
    return {
        "success": True,
        "status": "healthy"
    }


@app.post("/analyze")
async def analyze_test(
    officer_id: str = Form("OFFICER001"),
    latitude: Union[str, float] = Form(0.0),
    longitude: Union[str, float] = Form(0.0),
    accuracy_meters: Union[str, float] = Form(0.0),
    image: UploadFile = File(...)
) -> Dict[str, Any]:
    """Primary analysis pipeline executing image reading, SHA-256 calculation, detection, and color science."""
    try:
        # Validate and read exact raw bytes
        original_bytes = await image.read()
        if not original_bytes:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Uploaded image file is empty."}
            )

        # 1. SHA-256 calculation on exact bytes
        image_sha256 = calculate_sha256(original_bytes)
        logger.info(f"[ANALYZE] image received from {officer_id}, bytes: {len(original_bytes)}, SHA256: {image_sha256}")

        # 2. Decode image with OpenCV
        np_arr = np.frombuffer(original_bytes, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None or cv_image.size == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Failed to decode image. Please upload a valid JPEG/PNG."}
            )

        h_img, w_img = cv_image.shape[:2]
        logger.info(f"[ANALYZE] image size: {w_img}x{h_img}")

        # 3. Image Quality Analysis
        quality = calculate_image_quality(cv_image)

        # 4. Reference Swatch Detection
        ref_card = detect_reference_card(cv_image)
        logger.info(f"[ANALYZE] reference detection: {ref_card['detected']}, RGB: {ref_card['color']['rgb']}")

        # 5. Test Cassette Detection
        test_cassette = detect_test_cassette(cv_image)
        logger.info(f"[ANALYZE] cassette detection: {test_cassette['detected']}, rect: {test_cassette['rectangle_xyxy']}")

        # 6. Test Area Detection & Color Measurement
        test_area = detect_test_area(cv_image, test_cassette["rectangle_xyxy"])
        logger.info(f"[ANALYZE] test area detection: {test_area['detected']}, RGB: {test_area['color']['rgb']}")

        # 7. Colourimetric Comparison
        comparison = compare_colors(ref_card["color"], test_area["color"])
        logger.info(f"[ANALYZE] LAB distance: {comparison['lab_distance']}, RGB distance: {comparison['rgb_distance']}")

        # 8. Classification
        result_label, confidence, is_presumptive, reason = classify_result(comparison)
        comparison["reason"] = reason
        logger.info(f"[ANALYZE] final result: {result_label}, confidence: {confidence}%")

        # 9. Annotated Image Generation
        annotated_b64 = create_annotated_image(
            cv_image,
            ref_card["rectangle_xyxy"],
            ref_card["color"],
            test_cassette["rectangle_xyxy"],
            test_area["rectangle_xyxy"],
            test_area["color"]
        )

        # 10. Generate Test ID & Metadata
        test_id = f"TEST-{secrets.token_hex(6).upper()}"
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Parse GPS safely
        try:
            lat_f = float(latitude)
            long_f = float(longitude)
            acc_f = float(accuracy_meters)
        except (ValueError, TypeError):
            lat_f, long_f, acc_f = 0.0, 0.0, 0.0

        gps_data = {
            "latitude": lat_f,
            "longitude": long_f,
            "accuracy_meters": acc_f
        }

        digital_record = {
            "test_id": test_id,
            "officer_id": officer_id,
            "result": result_label,
            "confidence_percent": confidence,
            "timestamp_utc": timestamp_utc,
            "gps": gps_data,
            "sha256": image_sha256,
            "reference_color": ref_card["color"],
            "test_color": test_area["color"],
            "colourimetric_comparison": comparison
        }

        response_payload = {
            "success": True,
            "test_id": test_id,
            "result": result_label,
            "confidence_percent": confidence,
            "officer_id": officer_id,
            "timestamp_utc": timestamp_utc,
            "gps": gps_data,
            "sha256": image_sha256,
            "image_quality": quality,
            "reference_card": ref_card,
            "test_cassette": test_cassette,
            "test_area": test_area,
            "classification": comparison,
            "presumptive_result": is_presumptive,
            "requires_lab_confirmation": True,
            "scientific_limitation": "This is a presumptive field-test result and does not replace laboratory confirmatory testing.",
            "annotated_image_base64": annotated_b64,
            "digital_record": digital_record,
            "message": f"{result_label}. {reason}"
        }

        return response_payload

    except Exception as e:
        logger.error(f"[ANALYZE] Unexpected error during analysis: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"An error occurred while processing the test image: {str(e)}"
            }
        )
