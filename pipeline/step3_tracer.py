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

    # Frangi at multiple scales matching hypocotyl width
    # At 33px/mm, hypocotyls are ~30-80px wide → sigmas 5-25
    px_per_mm = getattr(config, 'PX_PER_MM_HINT', 33.0)
    width_min  = max(3,  int(px_per_mm * 0.5))   # ~0.5mm
    width_max  = max(10, int(px_per_mm * 2.0))   # ~2mm
    n_scales   = 6
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

def _cost_map(vessel: np.ndarray, col_center: int,
              col_half_width: int) -> np.ndarray:
    """
    Build cost map from vesselness within a column band around col_center.
    Cost = 1 - vesselness  (low cost along filaments).
    Outside the column band cost is infinity (path cannot escape sideways).
    """
    h, w = vessel.shape
    cost = np.full((h, w), fill_value=np.inf, dtype=np.float32)

    c0 = max(0, col_center - col_half_width)
    c1 = min(w, col_center + col_half_width + 1)

    cost[:, c0:c1] = 1.0 - vessel[:, c0:c1]
    # Small floor so path always has finite cost even on dark pixels
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
                    col_half_width: int = None) -> dict:
    """
    Trace one hypocotyl from a cotyledon click point to the boundary line.

    Args:
        img          : RGB image (full side crop)
        vessel       : precomputed Frangi vesselness map
        click_row    : row of the cotyledon click
        click_col    : col of the cotyledon click
        boundary_row : row of the user-drawn bottom boundary
        col_half_width : horizontal search width (px); default from config

    Returns dict with:
        path         : list of (row, col) — the skeleton path
        length_px    : arc length in pixels
        length_mm    : arc length in mm (if px_per_mm known)
        mask         : bool array with path pixels set
    """
    h, w = img.shape[:2]

    if col_half_width is None:
        px_per_mm     = getattr(config, 'PX_PER_MM_HINT', 33.0)
        col_half_width = int(px_per_mm * config.TRACER_COLUMN_WIDTH_MM)

    # Clamp inputs
    start_row = max(0, min(click_row, h - 1))
    start_col = max(0, min(click_col, w - 1))
    end_row   = max(start_row + 5, min(boundary_row, h - 1))

    cost = _cost_map(vessel, start_col, col_half_width)
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
              clicks: list,
              boundary_row: int,
              px_per_mm: float = None) -> list:
    """
    Trace all hypocotyls from a list of click points.

    Args:
        img          : RGB image
        clicks       : list of (row, col) cotyledon positions
        boundary_row : bottom boundary row
        px_per_mm    : calibration (stored into config for tracer)

    Returns list of trace result dicts (one per click), in click order.
    """
    if px_per_mm is not None:
        config.PX_PER_MM_HINT = px_per_mm

    print(f"\n[Tracer] Computing Frangi vesselness map...")
    vessel = compute_vesselness(img)
    print(f"[Tracer] Tracing {len(clicks)} hypocotyls to boundary row={boundary_row}")

    results = []
    for i, (cr, cc) in enumerate(clicks):
        result = trace_hypocotyl(img, vessel, cr, cc, boundary_row)
        result['label'] = i + 1
        result['click'] = (cr, cc)
        mm_str = f"{result['length_mm']:.2f}mm" if result['length_mm'] else "?mm"
        print(f"  [{i+1}] click=({cr},{cc})  "
              f"length={result['length_px']:.0f}px  {mm_str}")
        results.append(result)

    return results
