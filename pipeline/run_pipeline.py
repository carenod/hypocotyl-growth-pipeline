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
    )
    parser.add_argument("image_t1", help="Timepoint 1 image")
    parser.add_argument("image_t2", help="Timepoint 2 image")
    parser.add_argument("--out",      default=None,
                        help="Output directory "
                             "(default: data/results/<t1>_vs_<t2>/)")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip image alignment step")
    parser.add_argument("--debug",    action="store_true",
                        help="Show extra debug plots")
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

    div_x = div_x_t1

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 2 – Alignment  (t2 → t1 coordinate space)")
    print("="*55)
    import numpy as np
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt

    if not args.no_align:
        from step2_alignment import align_images
        landmark_path = str(qc_dir / f"alignment_landmarks_{t1_name}_vs_{t2_name}.png")
        crop_t2_aligned, H = align_images(crop_t1, crop_t2,
                                          debug=args.debug,
                                          save_path=landmark_path)
        print(f"  Alignment landmarks → {landmark_path}")
        print(f"  ↑ Open this to check which features were matched")
    else:
        crop_t2_aligned = crop_t2
        H = None
        div_x = (div_x_t1 + div_x_t2) // 2
        print(f"  Alignment skipped.")

    # ── FIX 3: Save aligned t2 image for inspection ──
    aligned_path = str(qc_dir / f"aligned_t2_{t2_name}.png")
    plt.figure(figsize=(12, 9))
    plt.imshow(crop_t2_aligned)
    plt.title(f"Aligned t2: {t2_name}\n(check this matches t1 layout)",
              fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(aligned_path, dpi=120)
    plt.close()
    print(f"  Aligned t2 saved → {aligned_path}")
    print(f"  ↑ Open this file to check alignment quality before proceeding")

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 3a – Annotate t1 (click cotyledon + root tip per seedling)")
    print("="*55)
    from step3_click_interface import (run_t1_session, run_t2_session,
                                       show_qc_window)
    from step3_tracer import trace_all

    config.PX_PER_MM_HINT = ppm_t1 or 33.0

    # ── T1 annotation ──
    pairs_t1 = run_t1_session(
        crop_t1,
        title=f"t1: {t1_path.name}  —  Click COTYLEDON then ROOT TIP per seedling",
        image_path=str(t1_path),
    )

    # Log which side each seedling is on
    for i, p in enumerate(pairs_t1):
        r, c  = p['top']
        side  = "Left (A)" if c < div_x else "Right (B)"
        print(f"  [{i+1}] top=({r},{c}) bot={p['bot']} → {side}")

    # ── Trace t1 ──
    print("\n  → Tracing hypocotyls in t1...")
    results_t1 = trace_all(crop_t1, pairs_t1, px_per_mm=ppm_t1)

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 3b – Verify t2 positions")
    print("="*55)

    # crop_t2_aligned is already warped into t1 coordinate space, so t1 click
    # positions map directly onto the aligned image — no transformation needed.
    suggested = [{'top': p['top'], 'bot': p['bot'],
                  'color': p['color']} for p in pairs_t1]

    pairs_t2 = run_t2_session(
        crop_t2_aligned,
        title=f"t2: {t2_path.name}  —  Verify / correct suggested positions",
        suggested_pairs=suggested,
        image_path=str(t2_path),
    )

    # ── Trace t2 ──
    print("\n  → Tracing hypocotyls in t2...")
    config.PX_PER_MM_HINT = ppm_t2 or 33.0
    results_t2 = trace_all(crop_t2_aligned, pairs_t2, px_per_mm=ppm_t2)

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 4 – QC overlay")
    print("="*55)

    qc_path = str(qc_dir / f"qc_{t1_name}_vs_{t2_name}.png")
    show_qc_window(
        img_t1         = crop_t1,
        img_t2_aligned = crop_t2_aligned,
        results_t1     = results_t1,
        results_t2     = results_t2,
        px_per_mm_t1   = ppm_t1 or 1.0,
        px_per_mm_t2   = ppm_t2 or 1.0,
        out_path       = qc_path,
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
        l_t2_px = r2['length_px']  if r2 else None
        l_t1_mm = r1['length_mm']
        l_t2_mm = r2['length_mm']  if r2 else None
        delta   = (l_t2_mm - l_t1_mm) if (l_t1_mm and l_t2_mm) else None
        growth  = (delta / l_t1_mm * 100) if (delta and l_t1_mm) else None

        rows.append({
            'genotype_side'   : side,
            'label'           : label,
            'length_t1_px'    : round(l_t1_px, 1),
            'length_t2_px'    : round(l_t2_px, 1)   if l_t2_px else None,
            'length_t1_mm'    : round(l_t1_mm, 3)   if l_t1_mm else None,
            'length_t2_mm'    : round(l_t2_mm, 3)   if l_t2_mm else None,
            'delta_mm'        : round(delta, 3)      if delta    else None,
            'growth_rate_pct' : round(growth, 1)     if growth   else None,
            'matched'         : r2 is not None,
            'notes'           : '',
        })

    df = pd.DataFrame(rows)
    print(f"\n  Measurements ({len(df)} hypocotyls):")
    print(df[['genotype_side', 'label', 'length_t1_mm',
              'length_t2_mm', 'delta_mm', 'growth_rate_pct']].to_string(index=False))

    xlsx_path = out_dir / f"measurements_{t1_name}_vs_{t2_name}.xlsx"
    export_excel(
        df, str(xlsx_path),
        px_per_mm_t1 = ppm_t1 or 1.0,
        px_per_mm_t2 = ppm_t2 or 1.0,
        image_t1     = str(t1_path),
        image_t2     = str(t2_path),
    )

    print("\n" + "="*55)
    print("  DONE")
    print("="*55)
    print(f"  Aligned t2  → {aligned_path}")
    print(f"  QC image    → {qc_path}")
    print(f"  Excel       → {xlsx_path}")
    print()


if __name__ == "__main__":
    main()
