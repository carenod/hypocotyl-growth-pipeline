#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  run_pipeline.py  –  main entry point
#
#  Run from the PROJECT ROOT:
#    python pipeline/run_pipeline.py data/raw/IMG_t1.JPG data/raw/IMG_t2.JPG
# ─────────────────────────────────────────────

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

PROJECT_ROOT = PIPELINE_DIR.parent


def main():
    parser = argparse.ArgumentParser(
        description="Arabidopsis hypocotyl growth measurement pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/run_pipeline.py data/raw/IMG_1721.JPG data/raw/IMG_1731.JPG
  python pipeline/run_pipeline.py data/raw/t1.JPG data/raw/t2.JPG --out data/results/exp01
  python pipeline/run_pipeline.py data/raw/t1.JPG data/raw/t2.JPG --no-align
        """
    )
    parser.add_argument("image_t1", help="Timepoint 1 image (e.g. data/raw/IMG_1721.JPG)")
    parser.add_argument("image_t2", help="Timepoint 2 image (e.g. data/raw/IMG_1731.JPG)")
    parser.add_argument("--out",      default=None,
                        help="Output directory (default: data/results/<t1>_vs_<t2>/)")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip image alignment step")
    parser.add_argument("--debug",    action="store_true",
                        help="Show extra debug plots during preprocessing")
    args = parser.parse_args()

    t1_path = Path(args.image_t1)
    t2_path = Path(args.image_t2)
    t1_name = t1_path.stem
    t2_name = t2_path.stem

    if args.out is None:
        out_dir = PROJECT_ROOT / "data" / "results" / f"{t1_name}_vs_{t2_name}"
    else:
        out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(exist_ok=True)

    print(f"\n  Project root : {PROJECT_ROOT}")
    print(f"  Image t1     : {t1_path}")
    print(f"  Image t2     : {t2_path}")
    print(f"  Output dir   : {out_dir}")

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 1 – Preprocessing")
    print("="*55)
    from step1_preprocess import preprocess
    import config

    crop_t1, roi_t1, ppm_t1, div_x_t1 = preprocess(str(t1_path), debug=args.debug)
    crop_t2, roi_t2, ppm_t2, div_x_t2 = preprocess(str(t2_path), debug=args.debug)

    print(f"\n  t1: {crop_t1.shape}  px/mm={ppm_t1}  divider x={div_x_t1}")
    print(f"  t2: {crop_t2.shape}  px/mm={ppm_t2}  divider x={div_x_t2}")

    # ══════════════════════════════════════════
    if not args.no_align:
        print("\n" + "="*55)
        print("  STEP 2 – Alignment  (t2 → t1 coordinate space)")
        print("="*55)
        from step2_alignment import align_images
        crop_t2_aligned, H = align_images(crop_t1, crop_t2, debug=args.debug)
        div_x = div_x_t1
        print(f"  Alignment done. Using t1 divider x={div_x}.")
    else:
        crop_t2_aligned = crop_t2
        H = None
        div_x = (div_x_t1 + div_x_t2) // 2
        print(f"\n  Alignment skipped. Average divider x={div_x}.")

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 3 – Click annotation + tracing")
    print("="*55)
    from step3_click_interface import run_click_session, show_qc_window
    from step3_tracer import trace_all
    import numpy as np
    import cv2

    # Store px/mm in config so tracer can use it for Frangi scales
    config.PX_PER_MM_HINT = ppm_t1 or 33.0

    print("\n  → Opening annotation window for t1...")
    print("  Instructions:")
    print("    1. Click LEFT then RIGHT edge of bottom boundary line")
    print("       (this separates hypocotyl from root)")
    print("    2. Click each cotyledon — a numbered dot appears")
    print("    3. Right-click a dot to undo the last click")
    print("    4. Press ENTER to confirm and run tracing")

    # Show the full plate crop for clicking (not split by side)
    clicks_img_coords, boundary_row = run_click_session(
        crop_t1,
        title=f"t1: {t1_path.name} — Annotate cotyledons"
    )

    if not clicks_img_coords:
        print("  No clicks recorded — exiting.")
        sys.exit(0)

    print(f"\n  {len(clicks_img_coords)} cotyledons annotated, "
          f"boundary at row {boundary_row}")

    # Determine which side each click belongs to
    for i, (r, c) in enumerate(clicks_img_coords):
        side = "Left (A)" if c < div_x else "Right (B)"
        print(f"    [{i+1}] row={r} col={c} → {side}")

    # ── Trace t1 ──
    print("\n  → Tracing hypocotyls in t1...")
    results_t1 = trace_all(
        crop_t1,
        clicks_img_coords,
        boundary_row,
        px_per_mm=ppm_t1,
    )

    # ── Map clicks to t2 via homography ──
    print("\n  → Mapping click points to t2...")
    if H is not None:
        # H maps t2→t1, so we need H_inv to map t1→t2
        H_inv = np.linalg.inv(H)
        clicks_t2 = []
        for r, c in clicks_img_coords:
            pt = np.array([[[float(c), float(r)]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(pt, H_inv)
            new_c, new_r = mapped[0][0]
            clicks_t2.append((int(new_r), int(new_c)))
    else:
        # No alignment — use same coordinates
        clicks_t2 = clicks_img_coords

    # ── Trace t2 ──
    print("\n  → Tracing hypocotyls in t2...")
    config.PX_PER_MM_HINT = ppm_t2 or 33.0
    results_t2 = trace_all(
        crop_t2_aligned,
        clicks_t2,
        boundary_row,
        px_per_mm=ppm_t2,
    )

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 4 – QC overlay")
    print("="*55)

    qc_path = str(qc_dir / f"qc_{t1_name}_vs_{t2_name}.png")
    show_qc_window(
        img_t1        = crop_t1,
        img_t2_aligned = crop_t2_aligned,
        results_t1    = results_t1,
        results_t2    = results_t2,
        boundary_row  = boundary_row,
        px_per_mm_t1  = ppm_t1 or 1.0,
        px_per_mm_t2  = ppm_t2 or 1.0,
        out_path      = qc_path,
    )

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 5 – Measurements & export")
    print("="*55)
    import pandas as pd
    from step6_export import export_excel

    rows = []
    for r1, r2 in zip(results_t1, results_t2):
        label = r1['label']
        col   = r1['click'][1]
        side  = "Left (A)" if col < div_x else "Right (B)"

        l_t1_px = r1['length_px']
        l_t2_px = r2['length_px'] if r2 else None
        l_t1_mm = r1['length_mm']
        l_t2_mm = r2['length_mm'] if r2 else None

        delta_mm    = (l_t2_mm - l_t1_mm) if (l_t1_mm and l_t2_mm) else None
        growth_pct  = (delta_mm / l_t1_mm * 100) if (delta_mm and l_t1_mm) else None
        matched     = r2 is not None

        rows.append({
            'genotype_side'   : side,
            'label'           : label,
            'length_t1_px'    : round(l_t1_px, 1),
            'length_t2_px'    : round(l_t2_px, 1) if l_t2_px else None,
            'length_t1_mm'    : round(l_t1_mm, 3) if l_t1_mm else None,
            'length_t2_mm'    : round(l_t2_mm, 3) if l_t2_mm else None,
            'delta_mm'        : round(delta_mm, 3) if delta_mm else None,
            'growth_rate_pct' : round(growth_pct, 1) if growth_pct else None,
            'matched'         : matched,
            'notes'           : '',
        })

    df = pd.DataFrame(rows)

    print(f"\n  Measurements ({len(df)} hypocotyls):")
    print(df[['genotype_side', 'label', 'length_t1_mm',
              'length_t2_mm', 'delta_mm', 'growth_rate_pct']].to_string(index=False))

    xlsx_path = out_dir / f"measurements_{t1_name}_vs_{t2_name}.xlsx"
    export_excel(
        df, str(xlsx_path),
        px_per_mm_t1=ppm_t1 or 1.0,
        px_per_mm_t2=ppm_t2 or 1.0,
        image_t1=str(t1_path),
        image_t2=str(t2_path),
    )

    print("\n" + "="*55)
    print("  DONE")
    print("="*55)
    print(f"  QC image → {qc_path}")
    print(f"  Excel    → {xlsx_path}")
    print()


if __name__ == "__main__":
    main()
