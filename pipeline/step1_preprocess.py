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


def _filter_peaks_consistent(peaks: np.ndarray, tolerance: float = 0.35) -> np.ndarray:
    """Keep only peaks connected by spacings within `tolerance` of the median."""
    if len(peaks) < 3:
        return peaks
    spacings   = np.diff(peaks)
    median_sp  = np.median(spacings)
    consistent = np.abs(spacings - median_sp) / (median_sp + 1e-6) < tolerance
    keep = np.zeros(len(peaks), dtype=bool)
    for i in range(len(spacings)):
        if consistent[i]:
            keep[i] = keep[i + 1] = True
    return peaks[keep]


def detect_px_per_mm(img: np.ndarray, plate_roi, debug: bool = False):
    """
    Detect ruler tick spacing and return px/mm.

    Uses the mean grayscale per row (one dip per tick, no Canny double-edges),
    a median-filter background subtraction, and simple peak finding.

    Returns (px_per_mm, display_peaks, ruler_strip).
    """
    from scipy.signal import find_peaks, savgol_filter
    from scipy.ndimage import median_filter as mfilt

    x, y, w, h = plate_roi
    img_h, img_w = img.shape[:2]
    ruler_x     = x + w
    ruler_strip = img[y: y + h, ruler_x: img_w]
    rs_h, rs_w  = ruler_strip.shape[:2]

    gray = cv2.cvtColor(ruler_strip, cv2.COLOR_RGB2GRAY).astype(float)

    # Use the central 70 % of the ruler width (avoids edges and number labels)
    c0 = max(0,    int(rs_w * 0.15))
    c1 = min(rs_w, int(rs_w * 0.85))
    row_mean = gray[:, c0:c1].mean(axis=1)

    # Background detrend: median filter over ~15 % of strip height
    bg_win = max(21, min(int(rs_h * 0.15), 301))
    bg_win = bg_win | 1   # must be odd
    background = mfilt(row_mean, size=bg_win)
    signal = background - row_mean   # positive = darker than local bg = tick

    # Light smoothing
    if rs_h > 15:
        signal = savgol_filter(signal, min(11, rs_h | 1), 3)

    # Find peaks, trying a range of prominence thresholds
    peaks = np.array([], dtype=int)
    for prom_frac in (0.15, 0.08, 0.04):
        peaks, _ = find_peaks(signal,
                               distance=10,
                               prominence=signal.max() * prom_frac)
        if len(peaks) >= 5:
            break

    if len(peaks) < 2:
        print("  [ruler] Not enough tick marks found.")
        return None, np.array([]), ruler_strip

    # Keep only the regularly-spaced chain
    peaks = _filter_peaks_consistent(peaks)

    if len(peaks) < 2:
        print("  [ruler] Ticks too irregular to calibrate.")
        return None, np.array([]), ruler_strip

    spacings       = np.diff(peaks)
    median_spacing = float(np.median(spacings))
    px_per_mm      = median_spacing / config.RULER_TICK_MM

    # Display the 12 peaks closest to the middle of the detected range
    mid   = len(peaks) // 2
    start = max(0, mid - 6)
    display_peaks = peaks[start: start + 12]

    print(f"  [ruler] {len(peaks)} ticks, "
          f"spacing = {median_spacing:.1f} px → {px_per_mm:.2f} px/mm")

    if debug:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(ruler_strip)
        for p in display_peaks:
            axes[0].axhline(p, color='cyan', linewidth=1.0, alpha=0.8)
        axes[0].set_title("Ruler — detected ticks (cyan)")
        axes[0].axis("off")
        axes[1].plot(signal, label='signal')
        axes[1].plot(peaks, signal[peaks], 'rx', label='peaks')
        axes[1].set_title("Detrended row profile")
        axes[1].set_xlabel("Row (px)")
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    return px_per_mm, display_peaks, ruler_strip


def verify_calibration(ruler_strip: np.ndarray, peaks: np.ndarray,
                       px_per_mm_auto, image_name: str = "") -> float:
    """
    Interactive calibration window.

    Shows the ruler strip zoomed to the detected tick region.
    Cyan lines = auto-detected ticks.

    To recalibrate manually:
      - Left-click to place/move line A or B (snaps to nearest detected tick)
      - Extra clicks move whichever of A/B is nearest
      - ↑/↓ adjust the mm distance assumed between A and B (default 10 mm)
      - Right-click or R to reset manual lines
      - Enter to accept
    """
    import matplotlib
    for _backend in ['TkAgg', 'Qt5Agg', 'WXAgg']:
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue

    rs_h, rs_w = ruler_strip.shape[:2]
    has_peaks  = len(peaks) > 0

    # ── initial y-limits: zoom to the tick region, not the full ruler ──
    if has_peaks:
        pad    = max(80, int((peaks[-1] - peaks[0]) * 0.08))
        y0_ini = max(0,    peaks[0]  - pad)
        y1_ini = min(rs_h, peaks[-1] + pad)
    else:
        y0_ini = rs_h // 3
        y1_ini = 2 * rs_h // 3

    state = {
        'px_per_mm' : px_per_mm_auto,
        'refs'      : [],    # 0, 1, or 2 y-positions of manual reference lines
        'ref_mm'    : 10,
        'confirmed' : False,
    }

    fig = plt.figure(figsize=(13, 9), facecolor='#0d0d1a')
    ax_r = fig.add_axes([0.03, 0.05, 0.54, 0.92])
    ax_i = fig.add_axes([0.62, 0.05, 0.36, 0.92])
    for ax in (ax_r, ax_i):
        ax.set_facecolor('#111122')
        ax.axis('off')

    ax_r.imshow(ruler_strip)
    ax_r.set_xlim(0, rs_w)
    ax_r.set_ylim(y1_ini, y0_ini)   # matplotlib y-axis is inverted for images

    title = f"Ruler — {image_name}\n" if image_name else "Ruler\n"
    ax_r.set_title(title +
                   "Cyan = detected ticks   Red A/B = your reference lines\n"
                   "Scroll = zoom   Click to set A/B   ↑/↓ = mm distance",
                   color='white', fontsize=9, pad=6)

    # Auto-detected ticks
    for p in peaks:
        ax_r.axhline(p, color='cyan', linewidth=1.0, alpha=0.55)

    ref_artists = []

    # ── helpers ─────────────────────────────────
    def _recompute():
        if len(state['refs']) == 2 and state['ref_mm'] > 0:
            dist_px = abs(state['refs'][1] - state['refs'][0])
            if dist_px > 0:
                state['px_per_mm'] = dist_px / state['ref_mm']

    def redraw_refs():
        for a in ref_artists:
            try: a.remove()
            except Exception: pass
        ref_artists.clear()
        labels = ('A', 'B')
        for i, ry in enumerate(state['refs']):
            ln  = ax_r.axhline(ry, color='#FF5050', linewidth=2.5, alpha=0.95,
                               zorder=6)
            lbl = ax_r.text(rs_w * 0.03, ry,
                            f' {labels[i]}', color='white',
                            fontsize=12, fontweight='bold', va='center',
                            bbox=dict(boxstyle='round,pad=0.25',
                                      facecolor='#FF5050', alpha=0.9),
                            zorder=7)
            ref_artists += [ln, lbl]
        fig.canvas.draw_idle()

    def redraw_info():
        ax_i.cla()
        ax_i.set_facecolor('#111122')
        ax_i.set_xlim(0, 1); ax_i.set_ylim(0, 1); ax_i.axis('off')

        ppm     = state['px_per_mm']
        source  = "manual" if state['refs'] else "auto-detected"
        ppm_str = f"{ppm:.2f}" if ppm else "n/a"

        lines = [
            ("CALIBRATION",              "#FFD700", True),
            ("",                         "white",   False),
            (f"px/mm  =  {ppm_str}",     "#64FFFF", True),
            (f"  ({source})",            "white",   False),
        ]
        if ppm:
            lines += [(f"  10 mm = {10 * ppm:.0f} px", "white", False)]

        lines += [
            ("",                         "white",   False),
            ("─" * 30,                   "#444",    False),
            ("",                         "white",   False),
            ("NAVIGATE:",                "#64FFFF", True),
            ("  Scroll = zoom",          "white",   False),
            ("",                         "white",   False),
            ("RECALIBRATE:",             "#FFFF64", True),
        ]

        nr = len(state['refs'])
        if nr == 0:
            lines += [
                ("  Click to place line A", "white", False),
            ]
        elif nr == 1:
            lines += [
                ("  Line A set",              "#FFAA40", True),
                ("  Click to place line B",   "white",   False),
            ]
        else:
            dist_px = abs(state['refs'][1] - state['refs'][0])
            lines += [
                (f"  A–B = {dist_px:.0f} px",            "white",   False),
                (f"  A–B = {state['ref_mm']} mm",         "#FFAA40", True),
                (f"  ↑/↓ to adjust mm count",             "white",   False),
                (f"  → px/mm = {ppm_str}",                "#64FF64", True),
            ]
            lines += [("  (click again to move A or B)", "#AAAAAA", False)]

        lines += [
            ("",                          "white",   False),
            ("  Right-click / R = reset", "#FFAAAA", False),
            ("",                          "white",   False),
            ("─" * 30,                    "#444",    False),
            ("",                          "white",   False),
            ("  ENTER = accept",          "#AAFFAA", True),
        ]

        n = len(lines)
        for j, (txt, col, bold) in enumerate(lines):
            ypos = 0.97 - j * (0.97 / (n + 1))
            ax_i.text(0.04, ypos, txt, color=col,
                      fontsize=9,
                      fontweight='bold' if bold else 'normal',
                      fontfamily='monospace', va='top',
                      transform=ax_i.transAxes, clip_on=True)
        fig.canvas.draw_idle()

    # ── event handlers ───────────────────────────
    def on_click(event):
        if event.inaxes != ax_r or state['confirmed']:
            return
        if event.button == 3:          # right-click → reset
            state['refs'].clear()
            state['ref_mm']    = 10
            state['px_per_mm'] = px_per_mm_auto
            redraw_refs(); redraw_info()
            return
        if event.button != 1 or event.ydata is None:
            return

        y = float(event.ydata)         # exact click position, no snap
        if len(state['refs']) < 2:
            state['refs'].append(y)
        else:
            # Move whichever of A / B is nearest to the click
            dA = abs(y - state['refs'][0])
            dB = abs(y - state['refs'][1])
            state['refs'][0 if dA <= dB else 1] = y

        _recompute()
        redraw_refs()
        redraw_info()

    def on_scroll(event):
        if event.inaxes != ax_r:
            return
        xd, yd = event.xdata, event.ydata
        if xd is None:
            return
        f  = 0.75 if event.button == 'up' else 1.0 / 0.75
        xl = ax_r.get_xlim()
        yl = ax_r.get_ylim()
        ax_r.set_xlim([xd + (v - xd) * f for v in xl])
        ax_r.set_ylim([yd + (v - yd) * f for v in yl])
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'enter':
            state['confirmed'] = True
            plt.close(fig)
        elif event.key == 'up' and len(state['refs']) == 2:
            state['ref_mm'] += 1
            _recompute(); redraw_info()
        elif event.key == 'down' and len(state['refs']) == 2 and state['ref_mm'] > 1:
            state['ref_mm'] -= 1
            _recompute(); redraw_info()
        elif event.key in ('r', 'R', 'escape'):
            state['refs'].clear()
            state['ref_mm']    = 10
            state['px_per_mm'] = px_per_mm_auto
            redraw_refs(); redraw_info()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event',       on_scroll)
    fig.canvas.mpl_connect('key_press_event',    on_key)

    redraw_info()
    plt.show(block=True)

    final = state['px_per_mm']
    if not final or final <= 0:
        final = px_per_mm_auto or config.PX_PER_MM_HINT
        print(f"  [ruler] Calibration not set – using fallback {final:.2f} px/mm")
    else:
        print(f"  [ruler] Calibration confirmed: {final:.2f} px/mm")
    return final


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

    px_per_mm_auto, peaks, ruler_strip = detect_px_per_mm(img, plate_roi, debug=debug)
    px_per_mm = verify_calibration(ruler_strip, peaks, px_per_mm_auto,
                                   image_name=Path(image_path).name)

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
