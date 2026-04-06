# ─────────────────────────────────────────────
#  step4_matching.py
#  • Match hypocotyl instances between t1 and t2
#    using centroid proximity (Hungarian algorithm)
#  • Both sides (left, right) matched independently
# ─────────────────────────────────────────────

import numpy as np
from scipy.optimize import linear_sum_assignment
import config


def match_instances(instances_t1: list, instances_t2: list,
                    max_distance: float = None) -> list:
    """
    Match instances from t1 to t2 by centroid distance.
    Only non-tangled instances are matched.
    Uses the Hungarian algorithm for optimal assignment.

    Returns list of match dicts:
      {
        'label'       : int  (1-indexed match number),
        't1'          : instance dict or None,
        't2'          : instance dict or None,
        'distance_px' : float,
        'matched'     : bool,
      }
    """
    if max_distance is None:
        max_distance = config.MAX_MATCH_DISTANCE_PX

    # Only match non-tangled instances
    good_t1 = [i for i in instances_t1 if not i['tangled']]
    good_t2 = [i for i in instances_t2 if not i['tangled']]

    if not good_t1 or not good_t2:
        print("  [matching] No valid instances to match on one or both sides")
        return []

    # Build cost matrix (centroid distances)
    n1, n2 = len(good_t1), len(good_t2)
    cost = np.full((n1, n2), fill_value=1e9)

    for i, inst1 in enumerate(good_t1):
        for j, inst2 in enumerate(good_t2):
            r1, c1 = inst1['centroid']
            r2, c2 = inst2['centroid']
            dist = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
            cost[i, j] = dist

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    label = 1
    matched_t2 = set()

    for i, j in zip(row_ind, col_ind):
        dist = cost[i, j]
        if dist <= max_distance:
            matches.append({
                'label'       : label,
                't1'          : good_t1[i],
                't2'          : good_t2[j],
                'distance_px' : dist,
                'matched'     : True,
            })
            matched_t2.add(j)
            label += 1
        else:
            # Too far apart – treat as unmatched
            matches.append({
                'label'       : label,
                't1'          : good_t1[i],
                't2'          : None,
                'distance_px' : dist,
                'matched'     : False,
            })
            label += 1

    # t2 instances with no t1 match
    for j, inst2 in enumerate(good_t2):
        if j not in matched_t2:
            matches.append({
                'label'       : label,
                't1'          : None,
                't2'          : inst2,
                'distance_px' : np.inf,
                'matched'     : False,
            })
            label += 1

    n_matched = sum(1 for m in matches if m['matched'])
    print(f"  [matching] {n_matched} matched pairs "
          f"(of {len(good_t1)} t1 / {len(good_t2)} t2 valid instances)")
    return matches


if __name__ == "__main__":
    # Quick smoke test with random centroids
    rng = np.random.default_rng(42)

    fake_t1 = [{'centroid': (rng.integers(50, 400), rng.integers(50, 200)),
                'tangled': False, 'id': i} for i in range(8)]
    fake_t2 = [{'centroid': (c[0] + rng.integers(-15, 15),
                             c[1] + rng.integers(-15, 15)),
                'tangled': False, 'id': i} for i, c in enumerate([f['centroid'] for f in fake_t1])]

    matches = match_instances(fake_t1, fake_t2)
    for m in matches:
        t1_id = m['t1']['id'] if m['t1'] else None
        t2_id = m['t2']['id'] if m['t2'] else None
        print(f"  Label {m['label']}: t1={t1_id} ↔ t2={t2_id}  dist={m['distance_px']:.1f}  matched={m['matched']}")
