"""
ObjectCount AI
==============
Image-processing-based object detection and counting built with classical
computer vision: Python, OpenCV, NumPy, Pandas and Streamlit.

Pipeline
    preprocessing -> thresholding -> morphology -> (optional) watershed
    -> contour analysis -> shape measurements -> size classification
    plus a specialised Hough Circle detector for coins / round objects.

Run:
    pip install -r requirements.txt
    streamlit run objectcount_ai.py
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime
from html import escape

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
APP_NAME = "ObjectCount AI"
FONT = cv2.FONT_HERSHEY_SIMPLEX

SEGMENTATION_METHODS = [
    "Background colour contrast",
    "Dark objects on light background",
    "Light objects on dark background",
    "Adaptive threshold",
]
TARGET_EVERYDAY = "Everyday objects"
TARGET_COINS = "Coins / circular objects"

# Annotation colours (RGB) per size class and for circles.
SIZE_COLORS = {"Small": (245, 158, 11), "Medium": (59, 130, 246), "Large": (16, 185, 129)}
CIRCLE_COLOR = (16, 185, 129)
FLAG_COLOR = (239, 68, 68)

CONTOUR_COLUMNS = [
    "Object ID", "Size class", "Contour area (px²)", "Box area (px²)", "Width (px)",
    "Height (px)", "Perimeter (px)", "Circularity", "Aspect ratio", "Solidity",
    "Centroid X (px)", "Centroid Y (px)", "Review flag",
]
CIRCLE_COLUMNS = [
    "Circle ID", "Center X (px)", "Center Y (px)", "Radius (px)",
    "Diameter (px)", "Area estimate (px²)",
]

# ----------------------------------------------------------------------------
# Computer-vision engine
# ----------------------------------------------------------------------------


def odd_kernel(v) -> int:
    v = int(v)
    return max(1, v if v % 2 else v + 1)


def segment_image(rgb, threshold_mode, blur, morph_size, iterations,
                  use_watershed=False, adaptive_block=31, adaptive_c=5,
                  distance_peak_ratio=0.42):
    """Return grayscale, cleaned binary mask and an optional watershed label map."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (odd_kernel(blur), odd_kernel(blur)), 0)

    if threshold_mode == "Dark objects on light background":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif threshold_mode == "Light objects on dark background":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_mode == "Adaptive threshold":
        block = max(3, odd_kernel(adaptive_block))
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block, adaptive_c)
    else:
        # Foreground = pixels whose LAB colour differs from the border-estimated background.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]], axis=0)
        background = np.median(border, axis=0)
        dist_colour = np.sqrt(np.sum((lab - background) ** 2, axis=2))
        dist_colour = np.clip(dist_colour, 0, 255).astype(np.uint8)
        dist_colour = cv2.GaussianBlur(dist_colour, (5, 5), 0)
        _, mask = cv2.threshold(dist_colour, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = odd_kernel(morph_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(iterations))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(iterations))

    watershed_labels = None
    if use_watershed and cv2.countNonZero(mask) > 0:
        # Marker-controlled watershed: each local maximum of the distance map is a seed.
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        if distance.max() > 0:
            peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            local_max = cv2.dilate(distance, peak_kernel)
            peak_cutoff = max(2.0, float(distance.max()) * float(distance_peak_ratio))
            peaks = ((distance >= local_max - 1e-3) & (distance >= peak_cutoff)
                     & (binary > 0)).astype(np.uint8) * 255
            n_seed_labels, seed_labels = cv2.connectedComponents(peaks)
            if n_seed_labels > 1:
                markers = np.zeros(binary.shape, np.int32)
                markers[binary == 0] = 1
                for seed_id in range(1, n_seed_labels):
                    markers[seed_labels == seed_id] = seed_id + 1
                elevation = np.zeros_like(distance, dtype=np.uint8)
                fg = binary > 0
                elevation[fg] = np.clip(255 * (1 - distance[fg] / float(distance.max())),
                                        0, 255).astype(np.uint8)
                markers = cv2.watershed(cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR), markers)
                watershed_labels = markers.copy()
    return gray, mask, watershed_labels


def extract_regions(mask, watershed_labels, min_area):
    """Return contours of candidate objects in stable reading order."""
    regions = []
    if watershed_labels is not None:
        for label_id in np.unique(watershed_labels):
            if label_id <= 1:
                continue
            region = (watershed_labels == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) >= min_area:
                    regions.append(c)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = [c for c in contours if cv2.contourArea(c) >= min_area]
    regions.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    return regions


def annotation_scale(shape) -> float:
    """Font scale that stays legible on both thumbnails and large photos."""
    return float(np.clip(max(shape[:2]) / 900.0, 0.45, 1.3))


def draw_label(img, text, org, color, scale):
    thick = max(1, int(round(scale * 2)))
    cv2.putText(img, text, org, FONT, scale, (15, 23, 42), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def analyze_contours(rgb, contours, small_limit, medium_limit):
    """Measure each contour, flag suspicious ones and draw the annotated image."""
    annotated = rgb.copy()
    scale = annotation_scale(rgb.shape)
    line = max(2, int(round(scale * 2.5)))
    metrics = []
    for idx, contour in enumerate(contours, 1):
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        box_area = float(w * h)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        m = cv2.moments(contour)
        cx = int(m["m10"] / m["m00"]) if m["m00"] else int(x + w / 2)
        cy = int(m["m01"] / m["m00"]) if m["m00"] else int(y + h / 2)
        metrics.append({
            "contour": contour, "area": area, "perimeter": perimeter, "box": (x, y, w, h),
            "box_area": box_area, "cx": cx, "cy": cy,
            "circ": 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0,
            "aspect": w / h if h else 0.0,
            "solidity": area / hull_area if hull_area > 0 else 0.0,
        })

    median_area = float(np.median([m["area"] for m in metrics])) if metrics else 0.0
    rows = []
    for idx, m in enumerate(metrics, 1):
        size = ("Small" if m["box_area"] < small_limit
                else "Medium" if m["box_area"] < medium_limit else "Large")
        if len(metrics) >= 3 and median_area > 0 and m["area"] > 1.7 * median_area:
            flag = "Possible merge"
        elif m["solidity"] < 0.85:
            flag = "Irregular shape"
        else:
            flag = "OK"
        color = FLAG_COLOR if flag != "OK" else SIZE_COLORS[size]
        x, y, w, h = m["box"]
        cv2.drawContours(annotated, [m["contour"]], -1, (15, 23, 42), line + 2, cv2.LINE_AA)
        cv2.drawContours(annotated, [m["contour"]], -1, color, line, cv2.LINE_AA)
        draw_label(annotated, f"#{idx}" + (" ?" if flag != "OK" else ""),
                   (x, max(int(22 * scale), y - 6)), color, scale)
        rows.append({
            "Object ID": idx, "Size class": size,
            "Contour area (px²)": round(m["area"], 2), "Box area (px²)": int(m["box_area"]),
            "Width (px)": w, "Height (px)": h, "Perimeter (px)": round(m["perimeter"], 2),
            "Circularity": round(m["circ"], 3), "Aspect ratio": round(m["aspect"], 3),
            "Solidity": round(m["solidity"], 3),
            "Centroid X (px)": m["cx"], "Centroid Y (px)": m["cy"], "Review flag": flag,
        })
    return annotated, pd.DataFrame(rows, columns=CONTOUR_COLUMNS)


def hough_circles(rgb, dp, min_dist, param1, param2, min_radius, max_radius):
    """Detect circles. Small images are enlarged internally; results are mapped back."""
    scale = 3.0 if max(rgb.shape[:2]) < 500 else 1.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=max(1, int(min_dist * scale)),
        param1=param1, param2=param2,
        minRadius=max(0, int(min_radius * scale)), maxRadius=max(0, int(max_radius * scale)),
    )
    output = rgb.copy()
    fs = annotation_scale(rgb.shape)
    line = max(2, int(round(fs * 2.5)))
    rows = []
    if circles is not None:
        circles = np.round(circles[0] / scale).astype(int)
        circles = sorted(circles.tolist(), key=lambda c: (c[1], c[0]))
        for i, (x, y, r) in enumerate(circles, 1):
            cv2.circle(output, (x, y), r, (15, 23, 42), line + 2, cv2.LINE_AA)
            cv2.circle(output, (x, y), r, CIRCLE_COLOR, line, cv2.LINE_AA)
            cv2.circle(output, (x, y), max(2, line), FLAG_COLOR, -1, cv2.LINE_AA)
            draw_label(output, f"#{i}", (max(0, x - r), max(int(18 * fs), y - r - 6)),
                       CIRCLE_COLOR, fs * 0.9)
            rows.append({
                "Circle ID": i, "Center X (px)": x, "Center Y (px)": y,
                "Radius (px)": r, "Diameter (px)": 2 * r,
                "Area estimate (px²)": round(float(np.pi * r * r), 1),
            })
    return output, pd.DataFrame(rows, columns=CIRCLE_COLUMNS)


def png_bytes(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def area_based_estimate(df: pd.DataFrame) -> int:
    """Heuristic count that splits oversized regions by the median object area."""
    if df.empty or len(df) < 3:
        return len(df)
    areas = df["Contour area (px²)"].to_numpy(dtype=float)
    med = float(np.median(areas))
    if med <= 0:
        return len(df)
    return int(sum(max(1, round(a / med)) for a in areas))


def make_demo_image() -> np.ndarray:
    """Synthetic photo of six coins on a textured table, used for the built-in demo."""
    h, w = 300, 420
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 222 - 18 * (xx / w) - 10 * (yy / h)
    img = np.stack([base, base - 4, base - 12], axis=2) + rng.normal(0, 2.2, (h, w, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    centers = [(78, 78), (172, 70), (268, 84), (352, 72), (118, 208), (258, 214)]
    radii = [24, 23, 25, 22, 24, 23]
    shadow = np.zeros((h, w), np.uint8)
    for (cx, cy), r in zip(centers, radii):
        cv2.circle(shadow, (cx + 5, cy + 6), r, 255, -1, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0).astype(np.float32) / 255.0
    img = (img * (1 - 0.35 * shadow[..., None])).astype(np.uint8)
    for (cx, cy), r in zip(centers, radii):
        cv2.circle(img, (cx, cy), r, (150, 112, 40), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r - 2, (205, 166, 78), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r - 6, (190, 150, 64), 1, cv2.LINE_AA)
        cv2.circle(img, (cx - r // 4, cy - r // 4), r // 3, (228, 196, 112), -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


# ----------------------------------------------------------------------------
# Cached pipelines
# ----------------------------------------------------------------------------


@st.cache_data(show_spinner="Analysing image…", max_entries=12)
def run_contour_pipeline(rgb, mode, method, blur, morph, iters, watershed, peak,
                         min_area, small_limit, medium_limit):
    t0 = time.perf_counter()
    effective_method, effective_blur, auto_fallback = method, blur, False
    result = None

    if mode == "Auto Detect":
        # Heuristic search over method/blur combinations. It cannot know the true count.
        candidates = []
        for m in SEGMENTATION_METHODS:
            for b in (3, 5, 7):
                g, mk, lb = segment_image(rgb, m, b, morph, iters, watershed, 31, 5, peak)
                cnts = extract_regions(mk, lb if watershed else None, min_area)
                ratio = float(np.count_nonzero(mk)) / mk.size
                if not (0.003 < ratio < 0.75) or not cnts:
                    continue
                score = -abs(ratio - 0.18) - max(0, len(cnts) - 35) * 0.015
                areas = np.array([cv2.contourArea(c) for c in cnts])
                score -= 0.2 * float(np.mean(areas < min_area * 1.5))
                candidates.append((score, m, b, g, mk, lb, cnts, ratio))
        if candidates:
            _, effective_method, effective_blur, gray, mask, labels, contours, ratio = max(
                candidates, key=lambda c: c[0])
            result = (gray, mask, labels, contours, ratio)
        else:
            auto_fallback = True

    if result is None:
        gray, mask, labels = segment_image(rgb, method, blur, morph, iters, watershed, 31, 5, peak)
        contours = extract_regions(mask, labels if watershed else None, min_area)
        result = (gray, mask, labels, contours, float(np.count_nonzero(mask)) / mask.size)

    gray, mask, labels, contours, ratio = result
    annotated, df = analyze_contours(rgb, contours, small_limit, medium_limit)
    return {
        "gray": gray, "mask": mask, "labels": labels, "annotated": annotated, "df": df,
        "foreground_ratio": ratio, "method": effective_method, "blur": effective_blur,
        "auto_fallback": auto_fallback, "ms": (time.perf_counter() - t0) * 1000,
    }


@st.cache_data(show_spinner="Detecting circles…", max_entries=12)
def run_hough_pipeline(rgb, dp, min_dist, p1, p2, rmin, rmax):
    t0 = time.perf_counter()
    annotated, df = hough_circles(rgb, dp, min_dist, p1, p2, rmin, rmax)
    return {"annotated": annotated, "df": df, "ms": (time.perf_counter() - t0) * 1000}


# ----------------------------------------------------------------------------
# Quality checks and reporting
# ----------------------------------------------------------------------------


def accuracy_pct(predicted: int, expected: int):
    if expected <= 0:
        return None
    return max(0.0, 100.0 * (1 - abs(predicted - expected) / expected))


def build_checks(target, total, fg_ratio, df, expected):
    """Return a list of (level, title, detail). level is ok | warn | info."""
    checks = []
    if expected > 0:
        if total == expected:
            checks.append(("ok", "Matches the known count", f"Detected {total} of {expected} expected objects."))
        else:
            checks.append(("warn", "Differs from the known count",
                           f"Detected {total}, expected {expected}. Adjust the detector settings and verify the overlay."))

    if target == TARGET_COINS:
        if total == 0:
            checks.append(("warn", "No circles found",
                           "Lower the accumulator threshold or widen the radius limits."))
        else:
            r = df["Radius (px)"].to_numpy(dtype=float)
            cv_r = float(r.std() / r.mean()) if r.mean() > 0 else 0.0
            if cv_r > 0.25:
                checks.append(("warn", "Circle sizes vary widely",
                               f"Radius spread is {cv_r:.0%} of the mean. Some detections may be false positives."))
            else:
                checks.append(("ok", "Consistent circle sizes",
                               f"Radius spread is {cv_r:.0%} of the mean."))
    else:
        if total == 0:
            checks.append(("warn", "No objects found",
                           "Lower the minimum contour area or try another segmentation method."))
        if fg_ratio < 0.01 or fg_ratio > 0.65:
            checks.append(("warn", "Extreme foreground coverage",
                           f"{fg_ratio:.1%} of the image is foreground. The mask may be inverted or noisy."))
        if total > 40:
            checks.append(("warn", "Many regions found", "Some may be noise or texture. Raise the minimum area."))
        if total == 1:
            checks.append(("warn", "Only one region", "Touching objects may have merged into one."))
        merged = int((df["Review flag"] != "OK").sum()) if total else 0
        if merged:
            est = area_based_estimate(df)
            checks.append(("warn", f"{merged} object(s) flagged for review",
                           f"Oversized or irregular regions may be touching objects. If objects are similar in size, "
                           f"the area-based estimate is {est}. Try enabling watershed."))
        elif total >= 3:
            checks.append(("ok", "Region shapes look consistent", "No oversized or irregular regions flagged."))

    if not any(level == "warn" for level, _, _ in checks):
        checks.append(("info", "No automatic warnings", "Visual verification is still recommended."))
    return checks


def build_report(name, shape, detector, total, elapsed_ms, expected, acc, settings, checks, df):
    return json.dumps({
        "app": APP_NAME,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "image": {"name": name, "width_px": int(shape[1]), "height_px": int(shape[0])},
        "detector": detector,
        "object_count": int(total),
        "expected_count": int(expected) if expected > 0 else None,
        "counting_accuracy_pct": round(acc, 2) if acc is not None else None,
        "processing_ms": round(elapsed_ms, 1),
        "settings": settings,
        "checks": [{"level": lv, "title": t, "detail": d} for lv, t, d in checks],
        "measurements": json.loads(df.to_json(orient="records")),
    }, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Interface helpers
# ----------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {
  --bg:#050816; --surface:rgba(13,20,43,.88); --ink:#edf2ff; --muted:#a6b4d4; --line:rgba(150,174,230,.18);
  --accent:#91aaff; --accent2:#b18cff; --accent-soft:rgba(124,156,255,.16);
  --ok:#6ee7b7; --ok-bg:rgba(16,185,129,.13); --warn:#fcd34d; --warn-bg:rgba(245,158,11,.13);
  --bad:#fca5a5; --bad-bg:rgba(239,68,68,.13); --info:#c5d2ef; --info-bg:rgba(110,140,200,.13);
}
html, body, .stApp, .stApp * {font-family:'DM Sans','Segoe UI',system-ui,sans-serif;}
.stApp {position:relative;isolation:isolate;min-height:100vh;overflow:hidden;background:radial-gradient(ellipse at 18% 0%,rgba(67,56,202,.22),transparent 34%),radial-gradient(ellipse at 85% 18%,rgba(14,116,214,.16),transparent 32%),linear-gradient(145deg,#050816 0%,#080d22 48%,#050816 100%);color:var(--ink);}
/* Animated starfield layers sit behind the app and never block clicks. */
.stApp:before,.stApp:after {content:"";position:fixed;inset:-25%;pointer-events:none;z-index:0;background-repeat:repeat;will-change:transform,opacity;}
.stApp:before {opacity:.7;background-image:radial-gradient(1px 1px at 25px 35px,rgba(255,255,255,.9) 98%,transparent),radial-gradient(1px 1px at 145px 95px,rgba(165,196,255,.85) 98%,transparent),radial-gradient(2px 2px at 85px 175px,rgba(255,255,255,.75) 98%,transparent),radial-gradient(1px 1px at 205px 225px,rgba(173,156,255,.8) 98%,transparent);background-size:260px 260px;animation:starDrift 75s linear infinite;}
.stApp:after {opacity:.4;background-image:radial-gradient(1.5px 1.5px at 60px 80px,rgba(255,255,255,.9) 98%,transparent),radial-gradient(1px 1px at 190px 40px,rgba(120,190,255,.9) 98%,transparent),radial-gradient(2px 2px at 130px 210px,rgba(190,170,255,.8) 98%,transparent);background-size:310px 310px;animation:starTwinkle 5s ease-in-out infinite alternate,starDrift 110s linear infinite reverse;}
@keyframes starDrift {from{transform:translate3d(0,0,0)}to{transform:translate3d(-130px,95px,0)}}
@keyframes starTwinkle {from{opacity:.2}to{opacity:.62}}
@media (prefers-reduced-motion:reduce) {.stApp:before,.stApp:after{animation:none!important}}
.block-container {position:relative;z-index:1;max-width:1500px;padding-top:1.25rem;padding-bottom:3.5rem;}
#MainMenu, footer {visibility:hidden;} header[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:rgba(7,12,30,.92);border-right:1px solid var(--line);backdrop-filter:blur(18px);}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:var(--ink);letter-spacing:-.02em;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:var(--muted);}
[data-testid="stFileUploader"] {background:rgba(20,31,62,.72);border:1px dashed rgba(145,170,255,.45);border-radius:14px;padding:.4rem;}
[data-testid="stExpander"] {border:1px solid var(--line);border-radius:13px;background:rgba(13,20,43,.82);box-shadow:0 8px 24px rgba(0,0,0,.16);backdrop-filter:blur(14px);}
[data-testid="stExpander"] summary {font-weight:600;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px!important;border-color:var(--line)!important;box-shadow:0 5px 20px rgba(24,43,83,.035);background:rgba(13,20,43,.72);backdrop-filter:blur(14px);}
[data-testid="stMetric"] {background:rgba(13,20,43,.86);border:1px solid var(--line);padding:1rem 1.1rem;border-radius:14px;box-shadow:0 8px 24px rgba(0,0,0,.16);}
[data-testid="stMetricLabel"] {color:var(--muted);font-weight:600;}
[data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif;font-weight:700;letter-spacing:-.04em;}
/* Hero */
.oc-header {position:relative;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:1.2rem;flex-wrap:wrap;padding:1.45rem 1.6rem;background:linear-gradient(115deg,#111d3c 0%,#253d7c 56%,#4a5de0 100%);border:1px solid rgba(255,255,255,.15);border-radius:22px;margin-bottom:1.15rem;box-shadow:0 16px 36px rgba(36,56,125,.17);}
.oc-header:after {content:'';position:absolute;width:290px;height:290px;right:10%;top:-190px;border:1px solid rgba(255,255,255,.16);border-radius:50%;box-shadow:0 0 0 30px rgba(255,255,255,.035),0 0 0 65px rgba(255,255,255,.025);pointer-events:none;}
.oc-brand {display:flex;align-items:center;gap:1rem;position:relative;z-index:1;}
.oc-title {margin:0;font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;letter-spacing:-.045em;color:#fff;line-height:1.15;}
.oc-sub {margin:.42rem 0 0;color:#dbe5ff;font-size:.92rem;max-width:560px;}
.oc-eyebrow {font-size:.68rem;text-transform:uppercase;letter-spacing:.16em;color:#bfcaff;font-weight:700;margin-bottom:.32rem;}
.oc-pills {display:flex;gap:.45rem;flex-wrap:wrap;position:relative;z-index:1;}
.oc-pill {font-size:.75rem;font-weight:600;padding:.4rem .7rem;border-radius:999px;background:rgba(255,255,255,.11);color:#f4f6ff;border:1px solid rgba(255,255,255,.2);backdrop-filter:blur(8px);}
/* Metadata */
.oc-meta {display:flex;gap:.55rem;flex-wrap:wrap;margin:.35rem 0 1rem;}
.oc-chip {font-size:.78rem;padding:.35rem .7rem;border-radius:9px;background:rgba(16,25,52,.84);border:1px solid var(--line);color:var(--muted);box-shadow:0 2px 8px rgba(24,43,83,.025);}
.oc-chip b {color:var(--ink);font-weight:700;}
/* KPI cards */
.oc-kpis {display:grid;grid-template-columns:1.25fr repeat(4,1fr);gap:.75rem;margin-bottom:1.25rem;}
@media (max-width:1100px) {.oc-kpis{grid-template-columns:repeat(3,1fr);}}
@media (max-width:700px) {.oc-kpis{grid-template-columns:1fr 1fr;}.oc-title{font-size:1.45rem;}.oc-header{padding:1.1rem;}}
.oc-kpi {position:relative;overflow:hidden;background:rgba(13,20,43,.88);border:1px solid var(--line);border-radius:16px;padding:1rem 1.05rem;box-shadow:0 5px 18px rgba(24,43,83,.045);transition:transform .18s ease,box-shadow .18s ease;}
.oc-kpi:hover {transform:translateY(-2px);box-shadow:0 10px 24px rgba(24,43,83,.09);}
.oc-kpi-label {font-size:.77rem;font-weight:600;color:var(--muted);letter-spacing:.01em;}
.oc-kpi-value {font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;letter-spacing:-.045em;line-height:1.35;color:var(--ink);font-variant-numeric:tabular-nums;}
.oc-kpi-sub {font-size:.74rem;color:var(--muted);line-height:1.4;}
.oc-kpi-accent {background:linear-gradient(140deg,#4264f5,#7958ee);border-color:transparent;box-shadow:0 10px 24px rgba(76,91,232,.22);}
.oc-kpi-accent:after {content:'';position:absolute;width:120px;height:120px;border:1px solid rgba(255,255,255,.22);border-radius:50%;right:-45px;top:-55px;box-shadow:0 0 0 18px rgba(255,255,255,.06);}
.oc-kpi-accent .oc-kpi-label,.oc-kpi-accent .oc-kpi-sub {color:#e2e7ff;}
.oc-kpi-accent .oc-kpi-value {position:relative;z-index:1;color:#fff;font-size:2.65rem;line-height:1.15;}
/* Panels and status */
.oc-panel-title {font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1rem;letter-spacing:-.02em;margin-bottom:.15rem;color:var(--ink);}
.oc-panel-note {font-size:.8rem;color:var(--muted);margin-bottom:.65rem;line-height:1.45;}
.oc-check {display:flex;gap:.7rem;align-items:flex-start;padding:.72rem .85rem;border-radius:12px;margin-bottom:.55rem;border:1px solid transparent;}
.oc-check .t {font-weight:700;font-size:.88rem;}.oc-check .d {font-size:.81rem;line-height:1.45;opacity:.92;}.oc-check .i {font-weight:700;width:1.2rem;text-align:center;}
.oc-ok {background:var(--ok-bg);color:var(--ok);border-color:rgba(110,231,183,.28);}.oc-warn {background:var(--warn-bg);color:var(--warn);border-color:rgba(252,211,77,.28);}.oc-info {background:var(--info-bg);color:var(--info);border-color:rgba(165,190,240,.22);}
/* Empty state */
.oc-steps {display:grid;grid-template-columns:repeat(3,1fr);gap:.85rem;margin:1.15rem 0;}
@media (max-width:800px) {.oc-steps{grid-template-columns:1fr;}}
.oc-step {background:rgba(13,20,43,.86);border:1px solid var(--line);border-radius:16px;padding:1.15rem 1.2rem;box-shadow:0 5px 18px rgba(24,43,83,.04);}
.oc-step .n {display:inline-flex;width:2rem;height:2rem;border-radius:10px;background:linear-gradient(140deg,#e8edff,#f0eaff);color:var(--accent);align-items:center;justify-content:center;font-weight:700;font-size:.88rem;margin-bottom:.75rem;}
.oc-step h4 {margin:0 0 .35rem;font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;}.oc-step p {margin:0;font-size:.85rem;line-height:1.5;color:var(--muted);}
/* Tabs and controls */
[data-baseweb="tab-list"] {gap:.35rem;border-bottom:1px solid var(--line);padding:.25rem .25rem 0;}
[data-baseweb="tab"] {font-weight:700!important;padding:.65rem .9rem!important;border-radius:10px 10px 0 0;color:var(--muted)!important;}
[data-baseweb="tab"][aria-selected="true"] {color:var(--accent)!important;background:rgba(124,156,255,.16);}
[data-baseweb="tab-highlight"] {background:var(--accent)!important;height:3px!important;border-radius:3px 3px 0 0;}
.stButton>button,[data-testid="stDownloadButton"]>button {border-radius:11px;border:1px solid var(--line);background:rgba(18,28,57,.95);color:var(--ink);font-weight:700;min-height:2.6rem;transition:all .15s ease;}
.stButton>button:hover,[data-testid="stDownloadButton"]>button:hover {border-color:#aebcff;color:var(--accent);box-shadow:0 4px 12px rgba(69,104,245,.10);}
.stButton>button[kind="primary"] {background:linear-gradient(120deg,#4568f5,#7659ee);border-color:transparent;color:#fff;box-shadow:0 5px 14px rgba(69,104,245,.2);}
.stButton>button[kind="primary"]:hover {background:linear-gradient(120deg,#3657df,#6544df);color:#fff;}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stImage"] img {border-radius:12px;}
[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:var(--line);}
.stApp p,.stApp label,.stApp [data-testid="stMarkdownContainer"] {color:inherit;}
.stApp [data-testid="stCaptionContainer"] {color:var(--muted);}
</style>
"""

LOGO_SVG = (
    '<svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true">'
    '<rect width="40" height="40" rx="10" fill="#1F5EFF"/>'
    '<g fill="none" stroke="#fff" stroke-width="2.2">'
    '<circle cx="14" cy="15" r="5.5"/><circle cx="27" cy="14" r="4.5"/><circle cx="20" cy="27" r="6"/></g>'
    '<circle cx="14" cy="15" r="1.4" fill="#fff"/><circle cx="27" cy="14" r="1.4" fill="#fff"/>'
    '<circle cx="20" cy="27" r="1.4" fill="#fff"/></svg>'
)


def render_header():
    st.markdown(
        f'<div class="oc-header"><div class="oc-brand">{LOGO_SVG}<div>'
        '<div class="oc-eyebrow">Computer vision workspace · FOCV IA3</div>'
        f'<p class="oc-title">{APP_NAME}</p>'
        '<p class="oc-sub">Detect, count and measure objects with classical computer vision.</p>'
        '</div></div><div class="oc-pills"><span class="oc-pill">⌁ OpenCV powered</span>'
        '<span class="oc-pill">⚡ Runs locally</span><span class="oc-pill">✓ No AI API</span></div></div>',
        unsafe_allow_html=True)


def kpi(label, value, sub="", accent=False):
    cls = "oc-kpi oc-kpi-accent" if accent else "oc-kpi"
    return (f'<div class="{cls}"><div class="oc-kpi-label">{escape(str(label))}</div>'
            f'<div class="oc-kpi-value">{escape(str(value))}</div>'
            f'<div class="oc-kpi-sub">{escape(str(sub))}</div></div>')


def render_checks(checks):
    icons = {"ok": "✓", "warn": "!", "info": "i"}
    html = "".join(
        f'<div class="oc-check oc-{lv}"><div class="i">{icons[lv]}</div><div>'
        f'<div class="t">{escape(t)}</div><div class="d">{escape(d)}</div></div></div>'
        for lv, t, d in checks)
    st.markdown(html, unsafe_allow_html=True)


def panel(title, note=""):
    st.markdown(f'<div class="oc-panel-title">{escape(title)}</div>'
                + (f'<div class="oc-panel-note">{escape(note)}</div>' if note else ""),
                unsafe_allow_html=True)


def style_chart(chart, height=240):
    return (chart.properties(height=height, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#5B6B82", titleColor="#5B6B82", gridColor="#E6EBF2",
                            domainColor="#DDE3EC", tickColor="#DDE3EC", labelFont="IBM Plex Sans",
                            titleFont="IBM Plex Sans")
            .configure_legend(labelColor="#5B6B82", titleColor="#5B6B82"))


def load_demo():
    st.session_state["demo"] = True
    st.session_state["target"] = TARGET_COINS
    st.session_state["expected"] = 6


def clear_demo():
    st.session_state["demo"] = False
    st.session_state["expected"] = 0


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------


def main():
    st.set_page_config(page_title=f"{APP_NAME} | Object counting", page_icon="🔎", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()

    # ---- Sidebar ----------------------------------------------------------
    with st.sidebar:
        st.subheader("Image")
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"],
                                    label_visibility="collapsed")
        if st.session_state.get("demo") and uploaded is None:
            st.caption("Using the built-in demo image (six coins).")
            st.button("Remove demo image", on_click=clear_demo, key="clear_demo")
        elif uploaded is None:
            st.button("Load demo image", on_click=load_demo, key="load_demo_side")

        st.subheader("Detection")
        processing_mode = st.radio("Mode", ["Auto Detect", "Manual Mode"],
                                   help="Auto Detect tests several segmentation settings and keeps the one "
                                        "with the most plausible mask. It does not know the true count.")
        detection_target = st.radio("What are you counting?", [TARGET_EVERYDAY, TARGET_COINS], key="target",
                                    help="Coins / circular objects uses the Hough Circle Transform for the "
                                         "main result and the dashboard count.")
        threshold_mode = st.selectbox("Segmentation method", SEGMENTATION_METHODS,
                                      disabled=processing_mode == "Auto Detect",
                                      help="Chosen automatically in Auto Detect mode.")
        expected = st.number_input("Known object count (optional)", min_value=0, max_value=10000, step=1,
                                   key="expected",
                                   help="Enter the true count, if you know it, to see counting accuracy. 0 = unknown.")

        with st.expander("Segmentation"):
            min_area = st.slider("Minimum contour area (px²)", 10, 10000, 250, 10)
            blur = st.select_slider("Gaussian blur kernel", options=[1, 3, 5, 7, 9], value=3)
            morph_size = st.slider("Morphology kernel", 1, 15, 3, 2)
            iterations = st.slider("Morphology iterations", 0, 3, 1)
            use_watershed = st.checkbox("Watershed: split touching objects", value=False)
            peak_ratio = st.slider("Watershed peak threshold", 0.20, 0.70, 0.42, 0.02,
                                   help="Lower may create extra seeds; higher may merge nearby seeds.")

        with st.expander("Size classes"):
            small_limit = st.slider("Small / Medium boundary (px²)", 500, 50000, 15000, 500)
            medium_limit = st.slider("Medium / Large boundary (px²)", 1000, 100000, 40000, 1000)
            st.caption("Size classes use bounding-box area in pixels, not physical units.")

        with st.expander("Circle detector (Hough)"):
            hough_dp = st.slider("Accumulator resolution (dp)", 1.0, 2.5, 1.2, 0.1)
            hough_min_dist = st.slider("Minimum centre distance (px)", 5, 200, 15, 1)
            hough_param1 = st.slider("Edge threshold", 20, 250, 100, 5)
            hough_param2 = st.slider("Accumulator threshold", 8, 100, 24, 1,
                                     help="Lower values find more circles, including false ones.")
            radius_min = st.slider("Minimum radius (px)", 0, 100, 10, 1)
            radius_max = st.slider("Maximum radius (px)", 0, 300, 27, 1)

    # ---- Load image ---------------------------------------------------------
    if uploaded is not None:
        try:
            rgb = np.array(ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read this image: {exc}. Try a PNG or JPEG file.")
            st.stop()
        image_name = uploaded.name
    elif st.session_state.get("demo"):
        rgb = make_demo_image()
        image_name = "demo_coins.png"
    else:
        st.markdown(
            '<div class="oc-steps">'
            '<div class="oc-step"><div class="n">1</div><h4>Add an image</h4>'
            '<p>Upload a photo in the sidebar, or load the demo image to see how it works.</p></div>'
            '<div class="oc-step"><div class="n">2</div><h4>Choose what to count</h4>'
            '<p>Everyday objects use contour analysis. Coins and round objects use the Hough Circle Transform.</p></div>'
            '<div class="oc-step"><div class="n">3</div><h4>Review and export</h4>'
            '<p>Check the overlay and quality report, then download the image, CSV or JSON report.</p></div></div>',
            unsafe_allow_html=True)
        st.button("Load demo image", on_click=load_demo, type="primary", key="load_demo_main")
        st.caption("Tip: plain backgrounds and diffuse lighting give the most reliable classical segmentation. "
                   "Start with well-separated objects before trying touching ones.")
        st.stop()

    # ---- Validate settings --------------------------------------------------
    if small_limit >= medium_limit:
        st.error("The Medium / Large boundary must be greater than the Small / Medium boundary.")
        st.stop()
    if radius_max != 0 and radius_max <= radius_min:
        st.error("Maximum radius must be greater than minimum radius (or 0 for no limit).")
        st.stop()
    if max(rgb.shape[:2]) > 3500:
        st.info("This is a large image, so processing may take a few seconds.")

    # ---- Run pipelines (cached) --------------------------------------------
    contour = run_contour_pipeline(rgb, processing_mode, threshold_mode, blur, morph_size, iterations,
                                   use_watershed, peak_ratio, min_area, small_limit, medium_limit)
    circle = run_hough_pipeline(rgb, hough_dp, hough_min_dist, hough_param1, hough_param2,
                                int(radius_min), int(radius_max))

    use_circles = detection_target == TARGET_COINS
    active = circle if use_circles else contour
    df, annotated, elapsed_ms = active["df"], active["annotated"], active["ms"]
    total = len(df)
    detector_name = "Hough Circle Transform" if use_circles else "Contour analysis"
    fg_ratio = contour["foreground_ratio"]
    acc = accuracy_pct(total, int(expected))
    checks = build_checks(detection_target, total, fg_ratio, df, int(expected))

    # ---- Summary: metadata + KPIs ------------------------------------------
    chips = [("Image", image_name), ("Resolution", f"{rgb.shape[1]} × {rgb.shape[0]} px"),
             ("Detector", detector_name)]
    if not use_circles:
        chips.append(("Segmentation", f"{contour['method']} (blur {contour['blur']})"))
    st.markdown('<div class="oc-meta">' + "".join(
        f'<span class="oc-chip">{escape(k)}: <b>{escape(str(v))}</b></span>' for k, v in chips) + "</div>",
        unsafe_allow_html=True)

    if use_circles:
        mid = ("Median diameter", f"{df['Diameter (px)'].median():.0f} px" if total else "–", "Across detected circles")
        third = ("Mean radius", f"{df['Radius (px)'].mean():.1f} px" if total else "–", "Original-image pixels")
    else:
        mid = ("Median area", f"{df['Contour area (px²)'].median():,.0f} px²" if total else "–", "Per object contour")
        third = ("Foreground", f"{fg_ratio:.1%}", "Share of image in the mask")
    acc_card = (("Counting accuracy", f"{acc:.0f}%", f"Against known count of {int(expected)}") if acc is not None
                else ("Counting accuracy", "Not set", "Enter a known count in the sidebar"))
    st.markdown(
        '<div class="oc-kpis">'
        + kpi("Objects counted", total, f"Selected detector: {detector_name}", accent=True)
        + kpi(*mid) + kpi(*third) + kpi("Processing time", f"{elapsed_ms:.0f} ms", "Computation, excluding cache hits")
        + kpi(*acc_card) + "</div>", unsafe_allow_html=True)

    if processing_mode == "Auto Detect" and not use_circles:
        if contour["auto_fallback"]:
            st.warning("Auto Detect found no plausible mask, so your manual segmentation settings were used.")
        else:
            st.info("Auto Detect scores candidate masks heuristically and does not know the correct count. "
                    "Verify the result visually.")

    tabs = st.tabs(["Results", "Measurements", "Processing lab", "Detector comparison", "About the method"])

    # ---- Results ------------------------------------------------------------
    with tabs[0]:
        left, right = st.columns(2, gap="medium")
        with left, st.container(border=True):
            panel("Original image")
            st.image(rgb, width="stretch")
        with right, st.container(border=True):
            panel("Detected objects", f"{total} found with {detector_name.lower()}")
            st.image(annotated, width="stretch")

        st.markdown("&nbsp;", unsafe_allow_html=True)
        c1, c2 = st.columns([1.1, 1], gap="medium")
        with c1:
            panel("Quality report", "Automatic checks. They support, but do not replace, visual verification.")
            render_checks(checks)
            if use_watershed and not use_circles:
                st.caption("Watershed is enabled. Inspect the mask and region count; it can under- or over-segment.")
        with c2:
            panel("Export")
            settings = {
                "mode": processing_mode, "target": detection_target, "segmentation": contour["method"],
                "min_area": min_area, "blur": contour["blur"], "morph_kernel": morph_size,
                "morph_iterations": iterations, "watershed": use_watershed,
                "hough": {"dp": hough_dp, "min_dist": hough_min_dist, "param1": hough_param1,
                          "param2": hough_param2, "min_radius": int(radius_min), "max_radius": int(radius_max)},
            }
            report = build_report(image_name, rgb.shape, detector_name, total, elapsed_ms,
                                  int(expected), acc, settings, checks, df)
            st.download_button("Download annotated image (PNG)", png_bytes(annotated),
                               "objectcount_annotated.png", "image/png")
            st.download_button("Download measurements (CSV)", df.to_csv(index=False).encode("utf-8"),
                               "object_measurements.csv", "text/csv")
            st.download_button("Download full report (JSON)", report.encode("utf-8"),
                               "objectcount_report.json", "application/json")

    # ---- Measurements ---------------------------------------------------------
    with tabs[1]:
        if df.empty:
            st.warning("No objects were detected. Adjust the detector settings in the sidebar.")
        else:
            panel("Measurement table", f"{total} objects. Click a column header to sort.")
            st.dataframe(df, width="stretch", hide_index=True)

            col_a, col_b = st.columns(2, gap="medium")
            if use_circles:
                chart_df = pd.DataFrame({"Diameter": df["Diameter (px)"]})
                hist = alt.Chart(chart_df).mark_bar(color="#1F5EFF", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("Diameter:Q", bin=alt.Bin(maxbins=12), title="Diameter (px)"),
                    y=alt.Y("count()", title="Circles"))
                spatial = alt.Chart(pd.DataFrame({
                    "x": df["Center X (px)"], "y": df["Center Y (px)"], "r": df["Radius (px)"], "id": df["Circle ID"]})
                ).mark_circle(color="#1F5EFF", opacity=0.55).encode(
                    x=alt.X("x:Q", title="X (px)", scale=alt.Scale(domain=[0, rgb.shape[1]])),
                    y=alt.Y("y:Q", title="Y (px)", scale=alt.Scale(domain=[rgb.shape[0], 0])),
                    size=alt.Size("r:Q", legend=None, scale=alt.Scale(range=[80, 900])),
                    tooltip=["id", "x", "y", "r"])
                with col_a, st.container(border=True):
                    panel("Diameter distribution")
                    st.altair_chart(style_chart(hist), theme=None)
                with col_b, st.container(border=True):
                    panel("Circle positions", "Marker size follows the radius.")
                    st.altair_chart(style_chart(spatial), theme=None)
                st.caption("Circle measurements are estimates from the Hough Circle Transform.")
            else:
                chart_df = pd.DataFrame({"Area": df["Contour area (px²)"]})
                hist = alt.Chart(chart_df).mark_bar(color="#1F5EFF", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("Area:Q", bin=alt.Bin(maxbins=12), title="Contour area (px²)"),
                    y=alt.Y("count()", title="Objects"))
                size_df = (df["Size class"].value_counts().reindex(["Small", "Medium", "Large"], fill_value=0)
                           .rename_axis("Size").reset_index(name="Objects"))
                sizes = alt.Chart(size_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("Size:N", sort=["Small", "Medium", "Large"], title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Objects:Q", title="Objects"),
                    color=alt.Color("Size:N", legend=None, scale=alt.Scale(
                        domain=["Small", "Medium", "Large"], range=["#F59E0B", "#3B82F6", "#10B981"])))
                with col_a, st.container(border=True):
                    panel("Area distribution")
                    st.altair_chart(style_chart(hist), theme=None)
                with col_b, st.container(border=True):
                    panel("Size classes", "Based on bounding-box area.")
                    st.altair_chart(style_chart(sizes), theme=None)

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Mean area", f"{df['Contour area (px²)'].mean():,.0f} px²")
                s2.metric("Median area", f"{df['Contour area (px²)'].median():,.0f} px²")
                s3.metric("Mean circularity", f"{df['Circularity'].mean():.2f}")
                s4.metric("Flagged for review", int((df["Review flag"] != "OK").sum()))
            st.caption("For formal evaluation, compare predicted counts with manually labelled ground truth across "
                       "many images. Counting accuracy = 100 × (1 − |predicted − actual| / actual).")

    # ---- Processing lab -----------------------------------------------------
    with tabs[2]:
        st.caption("Intermediate stages of the contour pipeline, shown for the segmentation settings in use.")
        a, b = st.columns(2, gap="medium")
        with a, st.container(border=True):
            panel("1. Grayscale and smoothing")
            st.image(contour["gray"], clamp=True, width="stretch")
        with b, st.container(border=True):
            panel("2. Binary mask", "After thresholding and morphology.")
            st.image(contour["mask"], clamp=True, width="stretch")
        c, d = st.columns(2, gap="medium")
        with c, st.container(border=True):
            panel("3. Foreground overlay")
            overlay = rgb.copy()
            sel = contour["mask"] > 0
            overlay[sel] = (0.55 * overlay[sel] + 0.45 * np.array([31, 94, 255])).astype(np.uint8)
            st.image(overlay, width="stretch")
        with d, st.container(border=True):
            panel("4. Watershed boundaries")
            if contour["labels"] is not None:
                boundary = rgb.copy()
                boundary[contour["labels"] == -1] = (239, 68, 68)
                st.image(boundary, width="stretch")
            else:
                st.info("Enable watershed in the sidebar (Segmentation) to inspect separated regions.")

    # ---- Detector comparison --------------------------------------------------
    with tabs[3]:
        st.caption("Both detectors run on the same image. The one selected in the sidebar drives the dashboard.")
        a, b = st.columns(2, gap="medium")
        with a, st.container(border=True):
            panel("Contour analysis", f"{len(contour['df'])} objects" + ("  (selected)" if not use_circles else ""))
            st.image(contour["annotated"], width="stretch")
        with b, st.container(border=True):
            panel("Hough Circle Transform", f"{len(circle['df'])} circles" + ("  (selected)" if use_circles else ""))
            st.image(circle["annotated"], width="stretch")
        diff = len(contour["df"]) - len(circle["df"])
        if diff == 0:
            st.success("Both detectors agree on the count. That increases confidence but is not proof.")
        else:
            st.warning(f"The detectors differ by {abs(diff)}. Hough only finds circles, while contour analysis "
                       "may merge touching objects or pick up background texture. Inspect both overlays.")
        if not circle["df"].empty:
            st.download_button("Download Hough results (CSV)", circle["df"].to_csv(index=False).encode("utf-8"),
                               "hough_circle_measurements.csv", "text/csv", key="dl_hough")

    # ---- About ------------------------------------------------------------------
    with tabs[4]:
        l, r = st.columns([1.2, 1], gap="large")
        with l:
            st.subheader("What it does")
            st.write(
                f"{APP_NAME} is an image-processing application for detecting and counting objects. It is built "
                "with Python, OpenCV, NumPy, Pandas and Streamlit, and combines image preprocessing, "
                "thresholding, morphological operations, contour analysis and the Hough Circle Transform. "
                "A dashboard presents detection results and measurements, and the data can be exported as "
                "an annotated image, a CSV table or a JSON report.")
            st.subheader("Pipeline")
            st.markdown(
                "1. **Preprocess:** convert to grayscale or LAB colour distance, then apply Gaussian smoothing.\n"
                "2. **Threshold:** Otsu, adaptive, or background-colour contrast.\n"
                "3. **Clean up:** morphological opening and closing.\n"
                "4. **Separate (optional):** distance transform seeds with marker-controlled watershed.\n"
                "5. **Measure:** contours, area, perimeter, circularity, solidity, aspect ratio, centroid.\n"
                "6. **Classify and review:** size classes and automatic quality flags.\n"
                "7. **Circle mode:** Hough Circle Transform for coins and other round objects.")
            st.subheader("Test result")
            st.write("Testing detected all six coins in the sample image using the specialised circle detector.")
        with r:
            st.subheader("Known limitations")
            st.warning(
                "Touching or overlapping objects, textured backgrounds, shadows and reflections can still cause "
                "errors. Watershed depends on good seed points. Size classes use pixel area, not physical "
                "dimensions. Circle mode finds circles only. Auto Detect is a heuristic, not a guarantee.")
            st.subheader("Roadmap")
            st.markdown(
                "- Better segmentation for complex backgrounds\n"
                "- Larger-scale evaluation against labelled ground truth\n"
                "- Improved separation of overlapping objects\n"
                "- Physical-unit calibration from a reference object")
            st.subheader("FOCV concepts used")
            st.write("Filtering, thresholding, mathematical morphology, distance functions, watershed "
                     "segmentation, connected components, contour analysis and the Hough Transform.")

    st.caption("FOCV IA3  ·  Classical image processing  ·  No external AI API used")


if __name__ == "__main__":
    main()
