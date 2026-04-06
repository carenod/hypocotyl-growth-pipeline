# ─────────────────────────────────────────────
#  step3_segmentation.py
#  Seeded region growing hypocotyl segmentation.
#
#  Core idea:
#    Use green cotyledon blobs as seed points.
#    From each seed, flood-fill downward through a low-threshold
#    binary mask. Only pixels reachable from a cotyledon are kept.
#    Background marks (X, text, frame corners) are never reachable
#    because they have no cotyledon seed — they are ignored entirely.
#
#  Fallback (etiolated seedlings — no green cotyledons):
#    Standard connected-component segmentation + top-bulge shape filter.
#
#  Pipeline:
#    1. Detect green cotyledon blobs (LAB-A channel)
#    2. Low threshold (LAB-L) → binary mask of all bright structures
#    3. For each cotyledon blob → flood-fill downward through binary mask
#    4. Each filled region = one seedling instance
#    5. Skeletonize → tangle detection → hypocotyl/root split → measure
# ─────────────────────────────────────────────

import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

from plantcv import plantcv as pcv
from skimage.measure import regionprops
from scipy import ndimage as ndi
import config

pcv.params.debug = "none"

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
TUNING_DIR   = os.path.join(PROJECT_ROOT, "data", "results", "tuning")


# ─────────────────────────────────────────────
#  Step 1: Cotyledon detection
# ─────────────────────────────────────────────

def detect_cotyledons(img: np.ndarray):
    """
    Find individual green cotyledon blobs using the LAB A channel.
    LAB-A: values below 128 = green. OpenCV uses 0-255 range.

    Returns:
      green_mask  : uint8 binary mask of green regions
      seeds       : list of (row, col) centroid per cotyledon blob
      n_green_px  : total green pixel count
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    a   = lab[:, :, 1].astype(np.int16)
    green = ((128 - a) > config.COTYLEDON_GREEN_THRESHOLD).astype(np.uint8) * 255

    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN,  k, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k, iterations=3)

    _, blob_labels = cv2.connectedComponents(green, connectivity=8)
    seeds = []
    for bid in range(1, blob_labels.max() + 1):
        blob = (blob_labels == bid)
        if blob.sum() < config.COTYLEDON_MIN_BLOB_PX:
            continue
        rows, cols = np.where(blob)
        seeds.append((float(rows.mean()), float(cols.mean())))

    n_green_px = int(green.sum() // 255)
    print(f"  [cotyledon] {n_green_px} green px → {len(seeds)} cotyledon seeds")
    return green, seeds, n_green_px


# ─────────────────────────────────────────────
#  Step 2: Binary mask (permissive threshold)
# ─────────────────────────────────────────────

def make_binary_mask(img: np.ndarray) -> np.ndarray:
    """
    Low threshold to capture all bright structures — hypocotyls, marks,
    everything. The seeded flood-fill will later select only the
    structures reachable from a cotyledon.
    """
    gray   = pcv.rgb2gray_lab(rgb_img=img, channel='l')
    binary = pcv.threshold.binary(
        gray_img=gray,
        threshold=config.PCV_THRESH,
        object_type='light',
    )
    # Light cleanup only — avoid closing across gaps between seedlings
    binary = pcv.fill(bin_img=binary, size=config.PCV_FILL_SIZE)
    close_k = np.ones((config.PCV_CLOSE_KSIZE, config.PCV_CLOSE_KSIZE), np.uint8)
    binary  = pcv.closing(gray_img=binary, kernel=close_k)
    binary  = pcv.fill(bin_img=binary, size=config.PCV_FILL_SIZE // 2)
    return binary


# ─────────────────────────────────────────────
#  Step 3: Seeded flood-fill
# ─────────────────────────────────────────────

def seeded_flood_fill(binary: np.ndarray, seeds: list,
                      img_shape: tuple) -> np.ndarray:
    """
    For each cotyledon seed, flood-fill downward through the binary mask.
    Each seed produces one labelled region.

    Strategy:
      - Dilate the binary mask slightly upward so seeds sitting just
        above the hypocotyl top can connect.
      - Use scipy label with a pre-seeded marker array (watershed-style).
      - Only pixels connected to a seed in the binary mask are kept.

    Returns an integer label array (0 = background, 1..N = instances).
    """
    h, w = img_shape[:2]

    # Expand binary upward slightly so cotyledon seeds (which sit just
    # above the hypocotyl) connect into the binary mask
    up_kernel  = np.zeros((config.SEED_UPWARD_DILATION * 2 + 1,
                           config.SEED_UPWARD_DILATION * 2 + 1), np.uint8)
    up_kernel[:config.SEED_UPWARD_DILATION + 1, :] = 1   # only upward
    binary_exp = cv2.dilate(binary, up_kernel, iterations=1)

    # Build seed marker image
    markers = np.zeros((h, w), dtype=np.int32)
    valid_seeds = []
    for i, (sr, sc) in enumerate(seeds):
        r, c = int(sr), int(sc)
        # Place seed in a small disk around the cotyledon centroid
        for dr in range(-config.SEED_RADIUS, config.SEED_RADIUS + 1):
            for dc in range(-config.SEED_RADIUS, config.SEED_RADIUS + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    markers[rr, cc] = i + 1
        valid_seeds.append(i + 1)

    # Only keep seeds that touch the binary mask (or its upward expansion)
    # Seeds sitting completely outside the mask produce empty regions
    touching = {}
    for label_id in valid_seeds:
        seed_pixels = (markers == label_id)
        if np.any(binary_exp[seed_pixels]):
            touching[label_id] = True

    # scipy label: flood-fill each seed through connected binary pixels
    # We do this seed-by-seed to handle overlapping search zones correctly
    result = np.zeros((h, w), dtype=np.int32)
    out_label = 1

    for label_id in valid_seeds:
        if label_id not in touching:
            print(f"    seed {label_id}: not touching binary mask — skipped")
            continue

        # Mask for this seed only
        seed_mask = (markers == label_id).astype(np.uint8)

        # Dilate seed into binary_exp to find connected region
        combined = cv2.bitwise_or(
            binary_exp,
            (seed_mask * 255).astype(np.uint8)
        )
        # Label connected components in the expanded binary
        _, cc_labels = cv2.connectedComponents(combined, connectivity=8)

        # Find which component(s) the seed pixels belong to
        seed_ccs = set(cc_labels[seed_mask > 0].tolist()) - {0}

        # Combine those components into one instance mask
        instance = np.zeros((h, w), dtype=bool)
        for cc_id in seed_ccs:
            instance |= (cc_labels == cc_id)

        # Only keep the part that is in the ORIGINAL (unexpanded) binary
        instance &= (binary > 0)

        if instance.sum() < config.MIN_AREA_PX:
            print(f"    seed {label_id}: region too small ({instance.sum()}px) — skipped")
            continue

        # Check for overlap with already-assigned pixels
        overlap = np.sum(instance & (result > 0))
        if overlap > instance.sum() * 0.5:
            print(f"    seed {label_id}: >50% overlap with existing instance — skipped")
            continue

        # Remove overlapping pixels (first-come, first-served)
        instance[result > 0] = False

        result[instance] = out_label
        out_label += 1

    n_found = result.max()
    print(f"  [seeded] {n_found} instances grown from {len(valid_seeds)} seeds")
    return result


# ─────────────────────────────────────────────
#  Step 4: Skeleton utilities
# ─────────────────────────────────────────────

def _axes(props):
    major = float(getattr(props, 'axis_major_length', None)
                  or getattr(props, 'major_axis_length', 0))
    minor = float(getattr(props, 'axis_minor_length', None)
                  or getattr(props, 'minor_axis_length', 0)) + 1e-6
    return major, minor


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    m    = (mask * 255).astype(np.uint8)
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
#  Step 5: Fallback (etiolated — no green)
# ─────────────────────────────────────────────

def has_top_bulge(mask: np.ndarray) -> bool:
    """
    Shape-based filter for etiolated seedlings.
    Hypocotyls widen at the top (seed attachment); marks don't.
    """
    rows = np.where(mask)[0]
    if len(rows) < 10:
        return False
    top_row = rows.min()
    bot_row = rows.max()
    height  = bot_row - top_row
    if height < 20:
        return False

    def width_at(f):
        row  = int(top_row + height * f)
        cols = np.where(mask[row, :])[0]
        return len(cols) if len(cols) > 0 else 0

    w_top    = np.mean([width_at(f) for f in [0.05, 0.10, 0.15]])
    w_upper  = np.mean([width_at(f) for f in [0.20, 0.25, 0.30]])
    w_middle = np.mean([width_at(f) for f in [0.45, 0.50, 0.55]])

    return (w_top   / (w_middle + 1e-6) > config.BULGE_TOP_RATIO_MIN or
            w_upper / (w_middle + 1e-6) > config.BULGE_UPPER_RATIO_MIN)


def fallback_segmentation(binary: np.ndarray) -> np.ndarray:
    """
    When no green cotyledons detected: standard connected-component
    segmentation filtered by shape + top-bulge test.
    Returns integer label array.
    """
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    n = labels.max()
    print(f"  [fallback] {n} raw components")
    out_label = 1
    result    = np.zeros_like(labels)

    for inst_id in range(1, n + 1):
        mask = (labels == inst_id)
        area = int(mask.sum())
        if area < config.MIN_AREA_PX or area > config.MAX_AREA_PX:
            continue
        props  = regionprops(mask.astype(np.uint8))[0]
        major, minor = _axes(props)
        if (major / minor) < config.MIN_ASPECT_RATIO:
            continue
        if props.solidity < config.MIN_SOLIDITY:
            continue
        if not has_top_bulge(mask):
            continue
        result[mask] = out_label
        out_label += 1

    print(f"  [fallback] {result.max()} instances passed shape+bulge filter")
    return result


# ─────────────────────────────────────────────
#  Step 6: Process instances
# ─────────────────────────────────────────────

def process_instances(labels: np.ndarray) -> list:
    instances = []
    for inst_id in range(1, labels.max() + 1):
        mask = (labels == inst_id)
        if mask.sum() < config.MIN_AREA_PX:
            continue

        props  = regionprops(mask.astype(np.uint8))[0]
        major, minor = _axes(props)
        aspect = major / minor

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

    green_mask, seeds, n_green_px = detect_cotyledons(img_crop)
    binary = make_binary_mask(img_crop)

    use_green = (n_green_px >= config.COTYLEDON_MIN_GREEN_PIXELS
                 and len(seeds) > 0)

    if use_green:
        labels = seeded_flood_fill(binary, seeds, img_crop.shape)
    else:
        print(f"  [segment] No green detected → fallback mode")
        labels = fallback_segmentation(binary)

    instances = process_instances(labels)
    return instances


# ─────────────────────────────────────────────
#  CLI tuning helper
#  Run from PROJECT ROOT:
#    python pipeline/step3_segmentation.py data/raw/IMG_1721.JPG
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)
    from step1_preprocess import preprocess

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python pipeline/step3_segmentation.py data/raw/<image.JPG>")
        sys.exit(1)

    os.makedirs(TUNING_DIR, exist_ok=True)
    print(f"[Tuning] Outputs → {TUNING_DIR}")

    crop, roi, ppm, div_x = preprocess(path, debug=False)
    left_crop  = crop[:, :div_x]
    right_crop = crop[:, div_x:]

    # ── Histogram ──
    gray = pcv.rgb2gray_lab(rgb_img=left_crop, channel='l')
    p50  = float(np.percentile(gray, 50))
    p75  = float(np.percentile(gray, 75))
    p90  = float(np.percentile(gray, 90))
    p95  = float(np.percentile(gray, 95))
    print(f"\n[Tuning] LAB-L  p50={p50:.0f}  p75={p75:.0f}  "
          f"p90={p90:.0f}  p95={p95:.0f}")
    print(f"[Tuning] Suggested PCV_THRESH range: {int(p75)}–{int(p90)}")

    fig0, ax0 = plt.subplots(figsize=(8, 4))
    ax0.hist(gray.ravel(), bins=128, color='steelblue', alpha=0.8)
    for pct, val, col in [(75,p75,'orange'),(90,p90,'red'),(95,p95,'darkred')]:
        ax0.axvline(val, color=col, linewidth=2, label=f'p{pct}={val:.0f}')
    ax0.set_title('LAB-L histogram'); ax0.set_xlabel('Pixel value'); ax0.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TUNING_DIR, 'plantcv_histogram.png'), dpi=100)
    plt.close()

    # ── Full pipeline on both sides with debug overlay ──
    palette = [(255,100,100),(100,255,100),(100,100,255),(255,255,100),
               (255,100,255),(100,255,255),(200,150,50),(50,200,150),
               (255,180,0),(0,200,180),(180,0,255),(0,100,255)]

    for side_name, side_crop in [("left", left_crop), ("right", right_crop)]:
        print(f"\n[Tuning] Processing {side_name} side...")
        green_mask, seeds, n_green_px = detect_cotyledons(side_crop)
        binary   = make_binary_mask(side_crop)
        use_green = (n_green_px >= config.COTYLEDON_MIN_GREEN_PIXELS
                     and len(seeds) > 0)

        if use_green:
            labels = seeded_flood_fill(binary, seeds, side_crop.shape)
        else:
            labels = fallback_segmentation(binary)

        instances = process_instances(labels)

        # Build overlay
        overlay = side_crop.copy()

        # Show green blobs
        overlay[green_mask > 0] = (
            overlay[green_mask > 0] * 0.4 + np.array([0, 220, 0]) * 0.6
        ).astype(np.uint8)

        # Draw seed positions and search radius
        for sr, sc in seeds:
            cv2.circle(overlay, (int(sc), int(sr)),
                       config.SEED_RADIUS, (0, 255, 255), 3)
            cv2.circle(overlay, (int(sc), int(sr)),
                       config.SEED_UPWARD_DILATION + 5, (0, 200, 200), -1)

        # Colour accepted instances
        for i, inst in enumerate(instances):
            c = palette[i % len(palette)] if not inst['tangled'] else (220, 50, 50)
            overlay[inst['mask']] = (
                overlay[inst['mask']] * 0.45 + np.array(c) * 0.55
            ).astype(np.uint8)
            r, col = int(inst['centroid'][0]), int(inst['centroid'][1])
            label_txt = str(i+1) if not inst['tangled'] else 'T'
            cv2.putText(overlay, label_txt, (col, r),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 3)

        n_ok     = len([i for i in instances if not i['tangled']])
        n_tg     = len([i for i in instances if i['tangled']])
        mode_str = "seeded" if use_green else "fallback"

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        axes[0].imshow(side_crop)
        axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title(f'Binary  threshold={config.PCV_THRESH}  '
                          f'({mode_str} mode)')
        axes[1].axis('off')
        axes[2].imshow(overlay)
        axes[2].set_title(
            f'{n_ok} accepted  {n_tg} tangled  [{mode_str} mode]\n'
            f'green=cotyledons  cyan=seed points  numbers=accepted')
        axes[2].axis('off')
        plt.tight_layout()
        out = os.path.join(TUNING_DIR, f'plantcv_current_config_{side_name}.png')
        plt.savefig(out, dpi=120); plt.close()
        print(f"[Tuning] {side_name} → {out}  "
              f"(accepted={n_ok}, tangled={n_tg})")

    print(f"\n[Tuning] Key parameters in config.py:")
    print(f"  PCV_THRESH                = {config.PCV_THRESH}"
          f"  ← lower to capture fainter hypocotyls")
    print(f"  PCV_FILL_SIZE             = {config.PCV_FILL_SIZE}"
          f"  ← noise removal before flood fill")
    print(f"  PCV_CLOSE_KSIZE           = {config.PCV_CLOSE_KSIZE}"
          f"  ← gap closing within filaments")
    print(f"  COTYLEDON_GREEN_THRESHOLD = {config.COTYLEDON_GREEN_THRESHOLD}"
          f"  ← lower if cotyledons not detected")
    print(f"  COTYLEDON_MIN_BLOB_PX     = {config.COTYLEDON_MIN_BLOB_PX}"
          f"  ← min cotyledon blob size")
    print(f"  SEED_RADIUS               = {config.SEED_RADIUS}"
          f"  ← seed point size around cotyledon centroid")
    print(f"  SEED_UPWARD_DILATION      = {config.SEED_UPWARD_DILATION}"
          f"  ← how far upward binary is dilated to connect seeds")
    print(f"  MAX_JUNCTION_POINTS       = {config.MAX_JUNCTION_POINTS}"
          f"  ← branch points above this → tangled")
