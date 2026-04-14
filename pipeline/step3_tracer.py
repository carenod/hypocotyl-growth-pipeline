# ─────────────────────────────────────────────
#  step3_tracer.py
#  Hypocotyl tracing using Frangi vesselness filter
#  + Dijkstra shortest-path from click to boundary line.
#
#  For each click point (cotyledon position):
#    1. Compute Frangi vesselness on a local column around the click
#    2. Build cost map: low cost where vesselness is high (= filament)
#    3. Find minimum-cost path from click downward to boundary row
#    4. That path = hypocotyl skeleton → measure arc length
# ─────────────────────────────────────────────

import numpy as np
import cv2
from skimage.filters import frangi
from skimage.color import rgb2gray
import heapq
import config


# ─────────────────────────────────────────────
#  Frangi vesselness map
# ─────────────────────────────────────────────

def compute_vesselness(img: np.ndarray) -> np.ndarray:
    """
    Apply Frangi filter to detect filament-like structures.
    Returns a float32 map in [0, 1] where high values = tubular structures.
    Works on brightness contrast, not absolute values — robust to
    uneven illumination and overlapping marks.
    """
    gray = rgb2gray(img).astype(np.float64)

    # Frangi sigma = expected half-width of the ridge.
    # Hypocotyls are typically 0.2–1.0 mm wide (thin plant stems).
    # sigma_min ≈ 0.1 mm half-width, sigma_max ≈ 0.5 mm half-width.
    px_per_mm = getattr(config, 'PX_PER_MM_HINT', 33.0)
    width_min  = max(2,  int(px_per_mm * 0.10))   # ~0.1 mm
    width_max  = max(6,  int(px_per_mm * 0.50))   # ~0.5 mm
    n_scales   = 5
    sigmas     = np.linspace(width_min, width_max, n_scales)

    vessel = frangi(
        gray,
        sigmas=sigmas,
        black_ridges=False,   # hypocotyls are bright on dark background
    ).astype(np.float32)

    # Normalise to [0, 1]
    vmax = vessel.max()
    if vmax > 0:
        vessel /= vmax

    return vessel


# ─────────────────────────────────────────────
#  Dijkstra shortest path downward
# ─────────────────────────────────────────────

def _cost_map(vessel: np.ndarray, c_min: int, c_max: int) -> np.ndarray:
    """
    Build cost map from vesselness within the column band [c_min, c_max].
    Cost = 1 - vesselness  (low cost along filaments).
    Outside the band cost is infinity.
    """
    h, w = vessel.shape
    cost = np.full((h, w), fill_value=np.inf, dtype=np.float32)
    c0 = max(0, c_min)
    c1 = min(w, c_max + 1)
    cost[:, c0:c1] = 1.0 - vessel[:, c0:c1]
    cost[:, c0:c1] = np.clip(cost[:, c0:c1], 0.05, 1.0)
    return cost


def dijkstra_downward(cost: np.ndarray,
                       start_row: int, start_col: int,
                       end_row: int) -> list:
    """
    Find minimum-cost path from (start_row, start_col) to any pixel
    in end_row, moving only downward (row can only increase).

    Allowed moves: down, down-left, down-right (no upward movement).
    Returns list of (row, col) tuples = the path.
    """
    h, w = cost.shape
    end_row = min(end_row, h - 1)

    # dist[r, c] = minimum cost to reach (r, c)
    dist = np.full((h, w), fill_value=np.inf, dtype=np.float32)
    dist[start_row, start_col] = 0.0

    # prev[r, c] = (prev_r, prev_c) for path reconstruction
    prev = {}

    # Min-heap: (cost, row, col)
    heap = [(0.0, start_row, start_col)]

    # Allowed moves: downward only
    moves = [(1, -1), (1, 0), (1, 1)]   # (dr, dc)

    while heap:
        d, r, c = heapq.heappop(heap)

        if r == end_row:
            # Reconstruct path
            path = []
            cur = (r, c)
            while cur in prev:
                path.append(cur)
                cur = prev[cur]
            path.append((start_row, start_col))
            path.reverse()
            return path

        if d > dist[r, c]:
            continue   # stale entry

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if np.isinf(cost[nr, nc]):
                continue
            new_dist = d + cost[nr, nc]
            if new_dist < dist[nr, nc]:
                dist[nr, nc] = new_dist
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_dist, nr, nc))

    # If end_row not reached, return straight vertical path as fallback
    return [(r, start_col) for r in range(start_row, end_row + 1)]


# ─────────────────────────────────────────────
#  Main tracing function
# ─────────────────────────────────────────────

def trace_hypocotyl(img: np.ndarray,
                    vessel: np.ndarray,
                    click_row: int, click_col: int,
                    boundary_row: int,
                    end_col: int = None,
                    col_half_width: int = None) -> dict:
    """
    Trace one hypocotyl from a cotyledon click to the root-tip click.

    Args:
        img          : RGB image (full side crop)
        vessel       : precomputed Frangi vesselness map
        click_row    : row of the cotyledon (top) click
        click_col    : col of the cotyledon (top) click
        boundary_row : row of the root-tip (bottom) click
        end_col      : col of the root-tip click (provides lateral guidance)
        col_half_width : extra horizontal margin beyond the click columns

    Returns dict with path, length_px, length_mm, mask.
    """
    h, w = img.shape[:2]

    px_per_mm = getattr(config, 'PX_PER_MM_HINT', 33.0)
    if col_half_width is None:
        col_half_width = int(px_per_mm * config.TRACER_COLUMN_WIDTH_MM)

    # Clamp inputs
    start_row = max(0, min(click_row,    h - 1))
    start_col = max(0, min(click_col,    w - 1))
    end_row   = max(start_row + 5, min(boundary_row, h - 1))
    if end_col is None:
        end_col = start_col
    end_col   = max(0, min(end_col, w - 1))

    # Column band covers both click columns plus a fixed margin
    c_min = min(start_col, end_col) - col_half_width
    c_max = max(start_col, end_col) + col_half_width
    cost  = _cost_map(vessel, c_min, c_max)

    # ── Lateral guidance ─────────────────────────────────────────────────
    # Add a small extra cost for pixels that deviate from the straight line
    # between the two clicks.  This steers the path toward the hypocotyl
    # without overriding the vesselness signal.
    n_rows  = end_row - start_row
    if n_rows > 0 and start_col != end_col:
        rows    = np.arange(h, dtype=float)
        t       = np.clip((rows - start_row) / n_rows, 0.0, 1.0)
        exp_col = start_col + t * (end_col - start_col)          # shape (h,)
        cols    = np.arange(w, dtype=float)
        lateral = np.abs(cols[np.newaxis, :] - exp_col[:, np.newaxis]) \
                  / max(col_half_width, 1)                        # shape (h, w)
        finite  = ~np.isinf(cost)
        cost[finite] += 0.3 * np.minimum(lateral[finite], 1.0)

    path = dijkstra_downward(cost, start_row, start_col, end_row)

    # Arc length (diagonal steps = sqrt(2))
    length_px = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i-1][0]
        dc = path[i][1] - path[i-1][1]
        length_px += np.sqrt(dr**2 + dc**2)

    px_per_mm  = getattr(config, 'PX_PER_MM_HINT', None)
    length_mm  = (length_px / px_per_mm) if px_per_mm else None

    # Build mask
    mask = np.zeros((h, w), dtype=bool)
    for r, c in path:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = True

    return {
        'path'      : path,
        'length_px' : length_px,
        'length_mm' : length_mm,
        'mask'      : mask,
        'start'     : (start_row, start_col),
        'end'       : path[-1] if path else (end_row, start_col),
    }


def trace_all(img: np.ndarray,
              pairs: list,
              px_per_mm: float = None) -> list:
    """
    Trace all hypocotyls from annotated pairs.

    Args:
        img      : RGB image
        pairs    : list of dicts {'top': (r,c), 'bot': (r,c), 'color': hex}
                   top = cotyledon position, bot = root tip (boundary per seedling)
        px_per_mm: calibration (stored into config for tracer)

    Returns list of trace result dicts (one per pair), in order.
    """
    if px_per_mm is not None:
        config.PX_PER_MM_HINT = px_per_mm

    print(f"\n[Tracer] Computing Frangi vesselness map...")
    vessel = compute_vesselness(img)
    print(f"[Tracer] Tracing {len(pairs)} hypocotyls...")

    results = []
    for i, pair in enumerate(pairs):
        cr,    cc    = pair['top']
        bot_r, bot_c = pair['bot']
        result = trace_hypocotyl(img, vessel, cr, cc, bot_r, end_col=bot_c)
        result['label'] = i + 1
        result['click'] = (cr, cc)
        result['color'] = pair.get('color', PALETTE[i % len(PALETTE)])
        mm_str = f"{result['length_mm']:.2f}mm" if result['length_mm'] else "?mm"
        print(f"  [{i+1}] top=({cr},{cc}) bot_row={bot_r}  "
              f"length={result['length_px']:.0f}px  {mm_str}")
        results.append(result)

    return results


# palette for fallback colour assignment
PALETTE = [
    '#FF6464', '#64FF64', '#6464FF', '#FFFF64',
    '#FF64FF', '#64FFFF', '#FFA040', '#40FFA0',
]
