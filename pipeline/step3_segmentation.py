# ─────────────────────────────────────────────
#  step3_segmentation.py
#  PlantCV-based hypocotyl segmentation.
# ─────────────────────────────────────────────

import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

from plantcv import plantcv as pcv
from skimage.measure import regionprops
import config

pcv.params.debug = "none"

# pipeline/ directory (where this file lives)
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
# project root = one level up
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
# tuning outputs go to data/results/tuning/ so they are git-ignored
TUNING_DIR   = os.path.join(PROJECT_ROOT, "data", "results", "tuning")


# ─────────────────────────────────────────────
#  Binary mask
# ─────────────────────────────────────────────

def make_binary_mask(img: np.ndarray) -> np.ndarray:
    """
    Binary mask of bright filaments (hypocotyls + roots) on dark background.

    Uses a simple absolute threshold on the LAB-L channel.
    The contrast between hypocotyls (~p90-p95 brightness) and background
    (~p50 brightness) is only ~20-30 gray levels, so absolute thresholding
    works better than adaptive (which gets confused by the low local contrast).
    """
    gray = pcv.rgb2gray_lab(rgb_img=img, channel='l')

    # Absolute threshold: keep pixels brighter than background + margin
    # config.PCV_THRESH is the absolute cutoff (0-255)
    binary = pcv.threshold.binary(
        gray_img=gray,
        threshold=config.PCV_THRESH,
        object_type='light',
    )

    # Remove small noise
    binary = pcv.fill(bin_img=binary, size=config.PCV_FILL_SIZE)

    # Close small gaps within filaments
    close_k = np.ones((config.PCV_CLOSE_KSIZE, config.PCV_CLOSE_KSIZE), np.uint8)
    binary  = pcv.closing(gray_img=binary, kernel=close_k)

    # Fill again after closing
    binary  = pcv.fill(bin_img=binary, size=config.PCV_FILL_SIZE // 2)

    return binary


# ─────────────────────────────────────────────
#  Instance labelling
# ─────────────────────────────────────────────

def label_instances(binary: np.ndarray) -> np.ndarray:
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    print(f"  [segment] {labels.max()} raw connected components")
    return labels


# ─────────────────────────────────────────────
#  Shape filter
# ─────────────────────────────────────────────

def _axes(props):
    major = float(getattr(props, 'axis_major_length', None)
                  or getattr(props, 'major_axis_length', 0))
    minor = float(getattr(props, 'axis_minor_length', None)
                  or getattr(props, 'minor_axis_length', 0)) + 1e-6
    return major, minor


def filter_by_shape(labels: np.ndarray) -> list:
    passed = []
    n = labels.max()
    verbose = n <= 30

    for inst_id in range(1, n + 1):
        mask = (labels == inst_id)
        area = int(mask.sum())

        if area < config.MIN_AREA_PX:
            if verbose:
                print(f"    id={inst_id}: SKIP area={area} < {config.MIN_AREA_PX}")
            continue
        if area > config.MAX_AREA_PX:
            if verbose:
                print(f"    id={inst_id}: SKIP area={area} > {config.MAX_AREA_PX}")
            continue

        props  = regionprops(mask.astype(np.uint8))[0]
        major, minor = _axes(props)
        aspect = major / minor

        if aspect < config.MIN_ASPECT_RATIO:
            if verbose:
                print(f"    id={inst_id}: SKIP AR={aspect:.1f} < {config.MIN_ASPECT_RATIO} (area={area})")
            continue
        if props.solidity < config.MIN_SOLIDITY:
            if verbose:
                print(f"    id={inst_id}: SKIP solidity={props.solidity:.2f} < {config.MIN_SOLIDITY}")
            continue

        passed.append((inst_id, mask, props, aspect))

    print(f"  [filter] {len(passed)} of {n} passed shape filter")
    return passed


# ─────────────────────────────────────────────
#  Skeleton utilities
# ─────────────────────────────────────────────

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask * 255).astype(np.uint8)
    skel = pcv.morphology.skeletonize(mask=m)
    if config.PCV_PRUNE_SIZE > 0:
        skel, _, _ = pcv.morphology.prune(
            skel_img=skel, size=config.PCV_PRUNE_SIZE, mask=m)
    return skel.astype(bool)


def count_branch_points(skel: np.ndarray) -> int:
    bp = pcv.morphology.find_branch_pts(skel_img=(skel * 255).astype(np.uint8))
    return int(np.sum(bp > 0))


def measure_arc_length(skel: np.ndarray) -> float:
    coords = np.column_stack(np.where(skel))
    if len(coords) < 2:
        return float(len(coords))
    total, visited, skel_set = 0.0, set(), set(map(tuple, coords))
    for r, c in coords:
        visited.add((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (r+dr, c+dc)
            if nb in skel_set and nb not in visited:
                total += 1.0
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nb = (r+dr, c+dc)
            if nb in skel_set and nb not in visited:
                total += np.sqrt(2)
    return total


def split_hypocotyl_root(skel: np.ndarray, fraction: float = None):
    if fraction is None:
        fraction = config.HYPOCOTYL_FRACTION
    rows = np.where(skel)[0]
    if len(rows) == 0:
        return skel.copy(), np.zeros_like(skel, dtype=bool), 0
    split_row = int(rows.min() + (rows.max() - rows.min()) * fraction)
    hypo = skel.copy(); hypo[split_row:, :] = False
    root = skel.copy(); root[:split_row, :] = False
    return hypo, root, split_row


# ─────────────────────────────────────────────
#  Per-instance processing
# ─────────────────────────────────────────────

def process_instances(shape_passed: list) -> list:
    instances = []
    for inst_id, mask, props, aspect in shape_passed:
        skel = skeletonize_mask(mask)
        if not np.any(skel):
            continue
        branch_pts = count_branch_points(skel)
        tangled    = branch_pts > config.MAX_JUNCTION_POINTS
        hypo_skel, root_skel, split_row = split_hypocotyl_root(skel)
        instances.append({
            'id'             : inst_id,
            'mask'           : mask,
            'skeleton'       : skel,
            'hypo_skel'      : hypo_skel,
            'root_skel'      : root_skel,
            'split_row'      : split_row,
            'centroid'       : props.centroid,
            'bbox'           : props.bbox,
            'length_px'      : measure_arc_length(hypo_skel),
            'full_length_px' : measure_arc_length(skel),
            'tangled'        : tangled,
            'junction_count' : branch_pts,
            'area'           : int(mask.sum()),
            'aspect_ratio'   : aspect,
        })
    kept    = [i for i in instances if not i['tangled']]
    tangled = [i for i in instances if i['tangled']]
    print(f"  [process] Accepted: {len(kept)}, Tangled: {len(tangled)}")
    return instances


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def segment_side(img_crop: np.ndarray, side_name: str = "") -> list:
    print(f"\n[Segmentation] Side: {side_name or 'unknown'}")
    binary    = make_binary_mask(img_crop)
    labels    = label_instances(binary)
    passed    = filter_by_shape(labels)
    instances = process_instances(passed)
    return instances


# ─────────────────────────────────────────────
#  CLI tuning helper
#  Run from the PROJECT ROOT:
#    python pipeline/step3_segmentation.py data/raw/IMG_1721.JPG
#  Saves diagnostic images to data/results/tuning/
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from step1_preprocess import preprocess
    import cv2

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python pipeline/step3_segmentation.py data/raw/<image.JPG>")
        sys.exit(1)

    # Create tuning output directory
    os.makedirs(TUNING_DIR, exist_ok=True)
    print(f"[Tuning] Outputs → {TUNING_DIR}")

    crop, roi, ppm, div_x = preprocess(path, debug=False)
    left_crop = crop[:, :div_x]
    gray = pcv.rgb2gray_lab(rgb_img=left_crop, channel='l')

    p50 = float(np.percentile(gray, 50))
    p75 = float(np.percentile(gray, 75))
    p90 = float(np.percentile(gray, 90))
    p95 = float(np.percentile(gray, 95))

    print(f"\n[Tuning] px/mm={ppm}  left_crop={left_crop.shape}")
    print(f"[Tuning] LAB-L  p50={p50:.0f}  p75={p75:.0f}  p90={p90:.0f}  p95={p95:.0f}")
    print(f"[Tuning] Background ≈ {p50:.0f}, hypocotyls likely above {p75:.0f}")
    print(f"[Tuning] Suggested threshold range: {int(p75)} – {int(p95)}")

    # ── Histogram ──
    fig0, ax0 = plt.subplots(figsize=(8, 4))
    ax0.hist(gray.ravel(), bins=128, color='steelblue', alpha=0.8)
    for pct, val, col in [(75, p75,'orange'), (90, p90,'red'), (95, p95,'darkred')]:
        ax0.axvline(val, color=col, linewidth=2, label=f'p{pct}={val:.0f}')
    ax0.set_title('LAB-L pixel histogram — background peak + bright filament tail')
    ax0.set_xlabel('Pixel value'); ax0.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TUNING_DIR, 'plantcv_histogram.png'), dpi=100); plt.close()

    # ── Absolute threshold grid ──
    thresholds = [
        int(p75), int(p75 + (p90-p75)*0.25), int(p75 + (p90-p75)*0.5),
        int(p90), int(p90 + (p95-p90)*0.5),  int(p95),
        int(p95 + 5), int(p95 + 10), int(p95 + 20),
    ]
    # pad to 12
    thresholds = sorted(set(thresholds))[:12]
    while len(thresholds) < 12:
        thresholds.append(thresholds[-1] + 5)

    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    for ax, thresh in zip(axes.flat, thresholds):
        b = pcv.threshold.binary(gray_img=gray, threshold=thresh, object_type='light')
        b = pcv.fill(bin_img=b, size=200)
        _, lbl = cv2.connectedComponents(b, connectivity=8)
        elongated = []
        for p in regionprops(lbl):
            if p.area < 1000:
                continue
            maj, mn = _axes(p)
            if maj / mn > 2.5:
                elongated.append(p)
        ax.imshow(b, cmap='gray')
        ax.set_title(f'threshold={thresh}\n{len(elongated)} elongated blobs (area>1000)')
        ax.axis('off')

    plt.suptitle(
        "Absolute threshold tuning (LAB-L channel)\n"
        "Find the value where hypocotyls are clean white filaments, "
        "not merged blobs\nThen set PCV_THRESH in config.py",
        fontsize=12)
    plt.tight_layout()
    grid_path = os.path.join(TUNING_DIR, 'plantcv_tuning_grid.png')
    plt.savefig(grid_path, dpi=100); plt.close()
    print(f"[Tuning] Grid saved → {grid_path}")

    # ── Current config overlay ──
    binary    = make_binary_mask(left_crop)
    labels    = label_instances(binary)
    passed    = filter_by_shape(labels)
    instances = process_instances(passed)

    overlay = left_crop.copy()
    palette = [(255,100,100),(100,255,100),(100,100,255),(255,255,100),
               (255,100,255),(100,255,255),(200,150,50),(50,200,150)]
    for i, inst in enumerate(instances):
        c = palette[i % len(palette)] if not inst['tangled'] else (220, 50, 50)
        overlay[inst['mask']] = (
            overlay[inst['mask']] * 0.5 + np.array(c) * 0.5
        ).astype(np.uint8)
        r, col = int(inst['centroid'][0]), int(inst['centroid'][1])
        lbl_txt = str(i + 1) if not inst['tangled'] else 'T'
        cv2.putText(overlay, lbl_txt, (col, r),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    n_ok = len([i for i in instances if not i['tangled']])
    n_tg = len([i for i in instances if i['tangled']])
    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 7))
    ax2[0].imshow(left_crop);  ax2[0].set_title('Original'); ax2[0].axis('off')
    ax2[1].imshow(binary, cmap='gray')
    ax2[1].set_title(f'Binary  threshold={config.PCV_THRESH}'); ax2[1].axis('off')
    ax2[2].imshow(overlay)
    ax2[2].set_title(f'{n_ok} accepted  {n_tg} tangled'); ax2[2].axis('off')
    plt.tight_layout()
    cfg_path = os.path.join(TUNING_DIR, 'plantcv_current_config.png')
    plt.savefig(cfg_path, dpi=120); plt.close()
    print(f"[Tuning] Current config → {cfg_path}  (accepted={n_ok}, tangled={n_tg})")
