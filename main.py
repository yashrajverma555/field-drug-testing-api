import base64
import hashlib
import io
import json
import math
import os
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Field Drug Testing Color Analysis API", version="3.0")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# IMPORTANT:
# These are DEMO prototypes only. Replace them with values measured from
# known-positive and known-negative images of YOUR exact test kit.
#
# The screenshot supplied by the user contains:
#   LEFT  = a large reference/color swatch
#   RIGHT = test cassette
# Therefore this version does NOT assume a 6-patch reference card.
DEMO_POSITIVE_RGB = np.array([180.0, 70.0, 60.0])
DEMO_NEGATIVE_RGB = np.array([215.0, 215.0, 215.0])

MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "90"))
MIN_MARGIN = float(os.getenv("MIN_MARGIN", "12"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "55"))

# Set DEMO_MODE=false after you have supplied real kit-specific prototypes.
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"


# ---------------------------------------------------------------------------
# BASIC COLOR FUNCTIONS
# ---------------------------------------------------------------------------
def rgb_hex(rgb):
    rgb = np.clip(np.asarray(rgb), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def rgb_to_hsv(rgb):
    arr = np.array([[np.clip(rgb, 0, 255).astype(np.uint8)]])
    return cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0, 0].astype(float)


def rgb_to_lab(rgb):
    arr = np.array([[np.clip(rgb, 0, 255).astype(np.uint8)]])
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(float)


def color_family(rgb):
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv

    if s < 25 and v > 220:
        return "WHITE"
    if s < 30 and v < 65:
        return "BLACK"
    if s < 35:
        return "GRAY"

    if h < 10 or h >= 170:
        return "RED"
    if h < 22:
        return "ORANGE"
    if h < 38:
        return "YELLOW"
    if h < 85:
        return "GREEN"
    if h < 105:
        return "CYAN"
    if h < 135:
        return "BLUE"
    if h < 165:
        return "PURPLE"
    return "RED"


def robust_color(roi_rgb):
    """
    Median RGB after trimming extreme brightness pixels.
    This is much less sensitive to glare, shadows and small dirt spots
    than a simple mean.
    """
    pixels = roi_rgb.reshape(-1, 3).astype(np.float32)

    if len(pixels) < 20:
        raise ValueError("ROI contains too few pixels")

    brightness = (
        0.2126 * pixels[:, 0]
        + 0.7152 * pixels[:, 1]
        + 0.0722 * pixels[:, 2]
    )

    p5, p95 = np.percentile(brightness, [5, 95])
    keep = (brightness >= p5) & (brightness <= p95)
    pixels = pixels[keep]

    if len(pixels) == 0:
        return np.median(roi_rgb.reshape(-1, 3), axis=0)

    return np.median(pixels, axis=0)


def measure_color(image_rgb, rect, label):
    x1, y1, x2, y2 = [int(v) for v in rect]
    h, w = image_rgb.shape[:2]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    # Ignore the border: sample only the middle 70%.
    rw = x2 - x1
    rh = y2 - y1
    sx1 = x1 + int(rw * 0.15)
    sy1 = y1 + int(rh * 0.15)
    sx2 = x2 - int(rw * 0.15)
    sy2 = y2 - int(rh * 0.15)

    roi = image_rgb[sy1:sy2, sx1:sx2]

    rgb = robust_color(roi)
    hsv = rgb_to_hsv(rgb)
    lab = rgb_to_lab(rgb)

    return {
        "label": label,
        "rectangle_xyxy": [sx1, sy1, sx2, sy2],
        "rgb": [round(float(x), 2) for x in rgb],
        "hex": rgb_hex(rgb),
        "hsv": [round(float(x), 2) for x in hsv],
        "lab": [round(float(x), 2) for x in lab],
        "brightness": round(float(np.mean(rgb)), 2),
        "saturation": round(float(hsv[1]), 2),
        "color_family": color_family(rgb),
        "pixel_count": int(roi.shape[0] * roi.shape[1]),
    }


# ---------------------------------------------------------------------------
# GEOMETRY / RECTANGLE DETECTION
# ---------------------------------------------------------------------------
def order_quad(points):
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    return np.array([
        pts[np.argmin(s)],       # top-left
        pts[np.argmin(d)],       # top-right
        pts[np.argmax(s)],       # bottom-right
        pts[np.argmax(d)],       # bottom-left
    ], dtype=np.float32)


def rect_from_quad(q):
    x1 = int(np.min(q[:, 0]))
    y1 = int(np.min(q[:, 1]))
    x2 = int(np.max(q[:, 0]))
    y2 = int(np.max(q[:, 1]))
    return (x1, y1, x2, y2)


def find_rectangles(image_rgb):
    """
    Finds large rectangular objects using multiple threshold levels.
    Returns candidates rather than blindly choosing the largest contour.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_sets = [
        cv2.Canny(blur, 30, 100),
        cv2.Canny(blur, 50, 150),
        cv2.Canny(blur, 80, 200),
    ]

    h, w = gray.shape
    image_area = h * w
    candidates = []

    for edges in edge_sets:
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < image_area * 0.015:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.035 * peri, True)

            if len(approx) != 4:
                x, y, cw, ch = cv2.boundingRect(contour)
                if cw * ch < image_area * 0.02:
                    continue
                q = np.array([
                    [x, y], [x + cw, y],
                    [x + cw, y + ch], [x, y + ch]
                ], dtype=np.float32)
            else:
                q = order_quad(approx.reshape(4, 2))

            rect = rect_from_quad(q)
            x1, y1, x2, y2 = rect
            rw = x2 - x1
            rh = y2 - y1

            if rw < 80 or rh < 80:
                continue

            aspect = rw / max(rh, 1)

            # Cards/cassettes are normally not extremely thin.
            if aspect < 0.35 or aspect > 3.5:
                continue

            fill = area / max(rw * rh, 1)

            candidates.append({
                "rect": rect,
                "quad": q.tolist(),
                "area": float(area),
                "fill": float(fill),
                "center": [
                    float((x1 + x2) / 2),
                    float((y1 + y2) / 2)
                ],
            })

    # De-duplicate near-identical rectangles.
    unique = []
    for c in sorted(candidates, key=lambda x: x["area"], reverse=True):
        x1, y1, x2, y2 = c["rect"]
        duplicate = False

        for u in unique:
            ux1, uy1, ux2, uy2 = u["rect"]

            ix1 = max(x1, ux1)
            iy1 = max(y1, uy1)
            ix2 = min(x2, ux2)
            iy2 = min(y2, uy2)

            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (
                (x2 - x1) * (y2 - y1)
                + (ux2 - ux1) * (uy2 - uy1)
                - inter
            )

            if union > 0 and inter / union > 0.75:
                duplicate = True
                break

        if not duplicate:
            unique.append(c)

    return unique


def detect_reference_and_cassette(image_rgb):
    """
    For the user's shown layout:
        reference card = left large rectangle
        cassette       = right large rectangle

    We score candidates by horizontal position, size and rectangularity.
    """
    h, w = image_rgb.shape[:2]
    candidates = find_rectangles(image_rgb)

    if not candidates:
        return None, None, candidates

    # Keep reasonably large objects.
    candidates = [
        c for c in candidates
        if (c["rect"][2] - c["rect"][0]) *
           (c["rect"][3] - c["rect"][1]) > 0.025 * w * h
    ]

    if not candidates:
        return None, None, []

    left = [
        c for c in candidates
        if c["center"][0] < 0.52 * w
    ]

    right = [
        c for c in candidates
        if c["center"][0] >= 0.48 * w
    ]

    def score(c, desired_side):
        cx = c["center"][0] / w
        cy = c["center"][1] / h

        side_score = (
            (1.0 - cx) if desired_side == "left" else cx
        )

        area_score = min(
            1.0,
            c["area"] / (0.12 * w * h)
        )

        vertical_score = 1.0 - min(abs(cy - 0.45), 0.45)

        return (
            0.50 * side_score
            + 0.30 * area_score
            + 0.20 * vertical_score
        )

    ref = max(
        left or candidates,
        key=lambda c: score(c, "left")
    )

    remaining = [
        c for c in candidates
        if c is not ref
    ]

    cassette = None

    if remaining:
        cassette = max(
            right or remaining,
            key=lambda c: score(c, "right")
        )

    return ref, cassette, candidates


# ---------------------------------------------------------------------------
# TEST AREA
# ---------------------------------------------------------------------------
def extract_test_area(image_rgb, cassette_rect):
    """
    The screenshot shows the reaction area approximately in the middle
    of the cassette. We deliberately avoid the cassette border.

    This is a geometry extractor, not a drug-result classifier.
    """
    x1, y1, x2, y2 = cassette_rect

    w = x2 - x1
    h = y2 - y1

    # Central reaction region.
    tx1 = x1 + int(w * 0.22)
    tx2 = x1 + int(w * 0.78)
    ty1 = y1 + int(h * 0.20)
    ty2 = y1 + int(h * 0.80)

    return (
        tx1, ty1, tx2, ty2
    )


# ---------------------------------------------------------------------------
# COLOR COMPARISON
# ---------------------------------------------------------------------------
def lab_distance(rgb_a, rgb_b):
    a = rgb_to_lab(rgb_a)
    b = rgb_to_lab(rgb_b)
    return float(np.linalg.norm(a - b))


def compare_with_reference(test_rgb, reference_rgb):
    """
    Measures whether the test region differs from the local reference.
    Useful for detecting a color reaction, but NOT sufficient by itself
    to establish a drug-positive result.
    """
    return {
        "rgb_difference": [
            round(float(test_rgb[i] - reference_rgb[i]), 2)
            for i in range(3)
        ],
        "absolute_rgb_difference": round(
            float(np.linalg.norm(test_rgb - reference_rgb)), 2
        ),
        "lab_distance": round(
            lab_distance(test_rgb, reference_rgb), 2
        ),
    }


def classify_demo(test_rgb):
    """
    Conservative prototype classifier.

    Uses LAB distance to kit-specific prototypes.
    Defaults are DEMO values and must be replaced with measurements from
    known samples of the exact kit.
    """
    d_pos = lab_distance(test_rgb, DEMO_POSITIVE_RGB)
    d_neg = lab_distance(test_rgb, DEMO_NEGATIVE_RGB)

    nearest = "POSITIVE" if d_pos < d_neg else "NEGATIVE"
    nearest_distance = min(d_pos, d_neg)
    margin = abs(d_pos - d_neg)

    # Confidence rises when the nearest class is close AND clearly
    # separated from the other class.
    separation = margin / max(d_pos + d_neg, 1.0)
    closeness = max(0.0, 1.0 - nearest_distance / 150.0)

    confidence = (
        100.0 * (0.65 * separation + 0.35 * closeness)
    )
    confidence = float(np.clip(confidence, 0, 99))

    if nearest_distance > MAX_DISTANCE:
        result = "INCONCLUSIVE"
        reason = (
            "Measured test color is outside the configured kit color "
            "range."
        )
    elif margin < MIN_MARGIN:
        result = "INCONCLUSIVE"
        reason = (
            "Positive and negative prototype colors are too close to "
            "the measured test color."
        )
    elif confidence < MIN_CONFIDENCE:
        result = "INCONCLUSIVE"
        reason = "Color separation is insufficient."
    else:
        result = nearest
        reason = (
            "Prototype color comparison passed the configured "
            "distance/separation checks."
        )

    return {
        "result": result,
        "confidence_percent": round(confidence, 1),
        "positive_distance_lab": round(d_pos, 2),
        "negative_distance_lab": round(d_neg, 2),
        "margin": round(margin, 2),
        "reason": reason,
        "prototype_mode": "DEMO" if DEMO_MODE else "KIT_SPECIFIC",
        "positive_prototype_rgb": DEMO_POSITIVE_RGB.tolist(),
        "negative_prototype_rgb": DEMO_NEGATIVE_RGB.tolist(),
    }


# ---------------------------------------------------------------------------
# IMAGE ANNOTATION
# ---------------------------------------------------------------------------
def annotate(
    image_rgb,
    reference_rect=None,
    cassette_rect=None,
    test_rect=None,
    result="INCONCLUSIVE",
    confidence=0,
):
    out = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    if reference_rect:
        x1, y1, x2, y2 = map(int, reference_rect)
        cv2.rectangle(
            out, (x1, y1), (x2, y2),
            (255, 0, 255), 4
        )
        cv2.putText(
            out,
            "REFERENCE",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if cassette_rect:
        x1, y1, x2, y2 = map(int, cassette_rect)
        cv2.rectangle(
            out, (x1, y1), (x2, y2),
            (0, 255, 0), 4
        )
        cv2.putText(
            out,
            "TEST CASSETTE",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if test_rect:
        x1, y1, x2, y2 = map(int, test_rect)
        cv2.rectangle(
            out, (x1, y1), (x2, y2),
            (0, 0, 255), 4
        )

        text = f"{result}  {confidence:.1f}%"

        cv2.putText(
            out,
            text,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def encode_jpeg(image_rgb):
    ok, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    if not ok:
        raise RuntimeError("Could not encode annotated image")

    return base64.b64encode(buffer.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# QUALITY CHECK
# ---------------------------------------------------------------------------
def image_quality(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # These are image-quality indicators only.
    score = 100.0

    if brightness < 45 or brightness > 235:
        score -= 25

    if contrast < 15:
        score -= 20

    if sharpness < 30:
        score -= 25

    score = float(np.clip(score, 0, 100))

    return {
        "score": round(score, 1),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "acceptable": score >= 50,
    }


# ---------------------------------------------------------------------------
# ANALYSIS PIPELINE
# ---------------------------------------------------------------------------
def analyze_image(
    image_bytes,
    officer_id,
    latitude,
    longitude,
    accuracy_meters,
):
    sha256 = hashlib.sha256(image_bytes).hexdigest()

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError("Uploaded file is not a readable image")

    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    quality = image_quality(image_rgb)

    reference_candidate, cassette_candidate, candidates = detect_reference_and_cassette(
        image_rgb
    )

    reference = reference_candidate["rect"] if isinstance(reference_candidate, dict) else reference_candidate
    cassette = cassette_candidate["rect"] if isinstance(cassette_candidate, dict) else cassette_candidate

    timestamp = datetime.now(timezone.utc).isoformat()
    test_id = "TEST-" + uuid.uuid4().hex[:12].upper()

    if reference is None or cassette is None:
        annotated = annotate(
            image_rgb,
            reference_rect=reference,
            cassette_rect=cassette,
            result="INCONCLUSIVE",
            confidence=0,
        )

        return {
            "success": True,
            "test_id": test_id,
            "result": "INCONCLUSIVE",
            "confidence_percent": 0.0,
            "officer_id": officer_id,
            "timestamp_utc": timestamp,
            "gps": {
                "latitude": latitude,
                "longitude": longitude,
                "accuracy_meters": accuracy_meters,
            },
            "sha256": sha256,
            "image_quality": quality,
            "reference": {
                "detected": reference is not None,
                "rectangle_xyxy": (
                    list(reference) if reference else None
                ),
            },
            "cassette": {
                "detected": cassette is not None,
                "rectangle_xyxy": (
                    list(cassette) if cassette else None
                ),
            },
            "test_area": None,
            "classification": {
                "result": "INCONCLUSIVE",
                "reason": (
                    "Could not confidently detect both the reference "
                    "swatch and test cassette."
                ),
            },
            "presumptive_result": False,
            "requires_lab_confirmation": True,
            "scientific_limitation": (
                "This is a prototype field-color analysis. It does not "
                "replace laboratory confirmatory testing."
            ),
            "annotated_image_base64": encode_jpeg(annotated),
        }

    # Measure reference swatch.
    reference_measurement = measure_color(
        image_rgb,
        reference,
        "REFERENCE_SWATCH",
    )

    # Determine reaction area from cassette.
    test_rect = extract_test_area(
        image_rgb,
        cassette,
    )

    test_measurement = measure_color(
        image_rgb,
        test_rect,
        "TEST_REACTION_AREA",
    )

    reference_rgb = np.array(
        reference_measurement["rgb"],
        dtype=float,
    )

    test_rgb = np.array(
        test_measurement["rgb"],
        dtype=float,
    )

    comparison = compare_with_reference(
        test_rgb,
        reference_rgb,
    )

    classification = classify_demo(test_rgb)

    # If the reference and reaction area are almost identical, report that
    # explicitly. This prevents a vague "inconclusive" from hiding the
    # actual measured color relationship.
    if comparison["lab_distance"] < 5:
        classification["reference_relationship"] = (
            "TEST AREA IS VERY CLOSE TO REFERENCE COLOR"
        )

    annotated = annotate(
        image_rgb,
        reference_rect=reference,
        cassette_rect=cassette,
        test_rect=test_rect,
        result=classification["result"],
        confidence=classification["confidence_percent"],
    )

    return {
        "success": True,
        "test_id": test_id,
        "result": classification["result"],
        "confidence_percent": classification["confidence_percent"],
        "officer_id": officer_id,
        "timestamp_utc": timestamp,
        "gps": {
            "latitude": latitude,
            "longitude": longitude,
            "accuracy_meters": accuracy_meters,
        },
        "sha256": sha256,
        "image_quality": quality,
        "reference": {
            "detected": True,
            "rectangle_xyxy": list(reference),
            "color": reference_measurement,
        },
        "cassette": {
            "detected": True,
            "rectangle_xyxy": list(cassette),
        },
        "test_area": {
            "rectangle_xyxy": list(test_rect),
            "color": test_measurement,
        },
        "comparison": comparison,
        "classification": classification,
        "presumptive_result": classification["result"] in {
            "POSITIVE",
            "NEGATIVE",
        },
        "requires_lab_confirmation": True,
        "scientific_limitation": (
            "This is a prototype presumptive field-test result and "
            "does not replace laboratory confirmatory testing. "
            "Positive/negative prototypes must be calibrated using "
            "known samples from the exact test kit."
        ),
        "detected_rectangle_count": len(candidates),
        "annotated_image_base64": encode_jpeg(annotated),
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Field Drug Testing Color Analysis API",
        "version": "3.0",
        "status": "online",
        "endpoint": "POST /analyze",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0",
        "demo_mode": DEMO_MODE,
    }


@app.post("/analyze")
async def analyze(
    officer_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    accuracy_meters: float = Form(...),
    image: UploadFile = File(...),
):
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image upload",
        )

    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is larger than 15 MB",
        )

    try:
        result = analyze_image(
            image_bytes=image_bytes,
            officer_id=officer_id,
            latitude=latitude,
            longitude=longitude,
            accuracy_meters=accuracy_meters,
        )

        return JSONResponse(content=result)

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
