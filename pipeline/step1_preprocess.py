# ─────────────────────────────────────────────
#  step1_preprocess.py
#  • Load image
#  • Detect and crop to plate interior
#  • Detect ruler strip → return px/mm ratio
# ─────────────────────────────────────────────

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_plate_roi(img: np.ndarray, debug: bool = False):
    """
    Find the bounding box of the plate interior.
    Tries multiple strategies and picks the best result.
    Returns (x, y, w, h) of the plate crop, ruler strip excluded.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray.shape

    # Ruler strip width to exclude from right edge
    ruler_strip_w = int(img_w * config.RULER_STRIP_FRACTION)
    # Work only on the non-ruler portion for plate detection
    gray_no_ruler = gray[:, : img_w - ruler_strip_w]
    img_no_ruler  = img[:,  : img_w - ruler_strip_w]
    work_w = img_w - ruler_strip_w

    best_roi = None
    best_score = 0

    # ── Strategy 1: edge contours (works when plate border is clear) ──
    blurred = cv2.GaussianBlur(gray_no_ruler, (7, 7), 0)
    edges = cv2.Canny(blurred, 20, 80)
    kernel = np.ones((11, 11), np.uint8)
    edges_d = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.05 * img_h * work_w:   # at least 5% of image
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / (h + 1e-6)
        if not (0.4 < aspect < 3.0):
            continue
        fill = area / (w * h + 1e-6)
        score = area * fill
        if score > best_score:
            best_score = score
            best_roi = (x, y, w, h)

    # ── Strategy 2: dark-region threshold (works when plate is dark) ──
    # Adaptive threshold based on image histogram
    p10 = np.percentile(gray_no_ruler, 10)
    p50 = np.percentile(gray_no_ruler, 50)
    thresh_val = min(p10 + (p50 - p10) * 0.5, 120)

    _, dark_mask = cv2.threshold(gray_no_ruler, int(thresh_val), 255, cv2.THRESH_BINARY_INV)
    kernel2 = np.ones((20, 20), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,  kernel2)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel2)
    contours2, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours2:
        area = cv2.contourArea(c)
        if area < 0.05 * img_h * work_w:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / (h + 1e-6)
        if not (0.4 < aspect < 3.0):
            continue
        score = area
        if score > best_score:
            best_score = score
            best_roi = (x, y, w, h)

    if best_roi is None or best_score < 0.05 * img_h * work_w:
        print("  [preprocess] Plate ROI not found – using full image minus ruler strip")
        # Still try to trim obvious white/foam border by looking for
        # the central dark region row/col extents
        margin_frac = 0.05
        mx = int(work_w  * margin_frac)
        my = int(img_h   * margin_frac)
        return mx, my, work_w - 2 * mx, img_h - 2 * my

    x, y, w, h = best_roi

    # Inward margin to exclude plate border itself
    margin = max(8, int(min(w, h) * 0.01))
    x = max(x + margin, 0)
    y = max(y + margin, 0)
    w = max(min(w - 2 * margin, work_w - x), 50)
    h = max(min(h - 2 * margin, img_h  - y), 50)

    if debug:
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 6)
        cv2.line(vis, (img_w - ruler_strip_w, 0),
                      (img_w - ruler_strip_w, img_h), (0, 255, 0), 4)
        plt.figure(figsize=(10, 7))
        plt.imshow(vis)
        plt.title("Plate ROI (blue) | Ruler boundary (green)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    print(f"  [preprocess] Plate ROI: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h


def detect_px_per_mm(img: np.ndarray, plate_roi, debug: bool = False) -> float:
    """
    Isolate the ruler strip on the right of the image.
    Detect horizontal tick marks and compute px/mm.
    Returns px_per_mm (float), or None if detection fails.
    """
    x, y, w, h = plate_roi
    img_h, img_w = img.shape[:2]

    # Ruler strip = the band to the right of the plate
    ruler_x = x + w
    ruler_strip = img[y: y + h, ruler_x: img_w]

    gray_ruler = cv2.cvtColor(ruler_strip, cv2.COLOR_RGB2GRAY)

    # Ruler ticks are bright marks on a pink/white background
    # Use Canny + horizontal line detection
    edges = cv2.Canny(gray_ruler, 30, 100)

    # Collapse horizontally – tick marks produce peaks in the row-sum profile
    row_profile = edges.sum(axis=1).astype(float)

    # Smooth the profile
    from scipy.signal import find_peaks, savgol_filter
    if len(row_profile) > 21:
        row_profile = savgol_filter(row_profile, 21, 3)

    # Find peaks (tick positions)
    peaks, _ = find_peaks(row_profile, distance=8, prominence=row_profile.max() * 0.05)

    if len(peaks) < 2:
        print("  [ruler] Could not detect enough tick marks. Px/mm ratio not computed.")
        return None

    # Median spacing between consecutive ticks
    spacings = np.diff(peaks)
    median_spacing = np.median(spacings)
    px_per_mm = median_spacing / config.RULER_TICK_MM

    print(f"  [ruler] Detected {len(peaks)} ticks, median spacing = {median_spacing:.1f} px → {px_per_mm:.2f} px/mm")

    if debug:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(ruler_strip)
        for p in peaks:
            axes[0].axhline(p, color='cyan', linewidth=0.8, alpha=0.7)
        axes[0].set_title("Ruler strip with detected ticks")
        axes[0].axis("off")

        axes[1].plot(row_profile)
        axes[1].plot(peaks, row_profile[peaks], "rx")
        axes[1].set_title("Row profile & peaks")
        axes[1].set_xlabel("Row (px)")
        plt.tight_layout()
        plt.show()

    return px_per_mm


def detect_dividing_line(img_crop: np.ndarray, debug: bool = False) -> int:
    """
    Detect the vertical dividing line in the middle of the plate.
    Returns the x-coordinate of the line within the cropped plate image.
    """
    gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Search only in the middle third of the image width
    search_x0 = w // 3
    search_x1 = 2 * w // 3
    strip = gray[:, search_x0:search_x1]

    # Detect vertical edges
    edges = cv2.Canny(strip, 30, 100)

    # Column-sum profile
    col_profile = edges.sum(axis=0).astype(float)

    from scipy.signal import find_peaks, savgol_filter
    if len(col_profile) > 11:
        col_profile = savgol_filter(col_profile, 11, 3)

    peaks, props = find_peaks(col_profile, distance=20, prominence=col_profile.max() * 0.1)

    if len(peaks) == 0:
        # Fallback: image center
        dividing_x = w // 2
        print(f"  [divider] Line not detected – using center ({dividing_x}px)")
        return dividing_x

    # Pick the peak closest to center of the search strip
    strip_center = (search_x1 - search_x0) // 2
    best = peaks[np.argmin(np.abs(peaks - strip_center))]
    dividing_x = search_x0 + best

    print(f"  [divider] Dividing line detected at x = {dividing_x} px")

    if debug:
        vis = img_crop.copy()
        cv2.line(vis, (dividing_x, 0), (dividing_x, h), (255, 255, 0), 3)
        plt.figure(figsize=(8, 5))
        plt.imshow(vis)
        plt.title("Detected dividing line (yellow)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return dividing_x


def preprocess(image_path: str, debug: bool = False):
    """
    Full preprocessing for one image.
    Returns:
      img_crop   : cropped plate RGB image (ruler excluded)
      plate_roi  : (x, y, w, h) in original image
      px_per_mm  : float or None
      dividing_x : int, x-coord of genotype divider within img_crop
    """
    print(f"\n[Preprocess] {Path(image_path).name}")
    img = load_image(image_path)

    plate_roi = detect_plate_roi(img, debug=debug)
    x, y, w, h = plate_roi
    img_crop = img[y: y + h, x: x + w]

    px_per_mm = detect_px_per_mm(img, plate_roi, debug=debug)
    dividing_x = detect_dividing_line(img_crop, debug=debug)

    return img_crop, plate_roi, px_per_mm, dividing_x


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/IMG_1714.JPG"
    crop, roi, ppm, div_x = preprocess(path, debug=True)
    print(f"Plate ROI: {roi}")
    print(f"Px/mm: {ppm}")
    print(f"Dividing line x: {div_x}")
    print(f"Cropped image shape: {crop.shape}")
