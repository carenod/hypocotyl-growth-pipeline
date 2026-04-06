#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  run_pipeline.py
#  Main entry point – processes one image pair
#
#  Run from the PROJECT ROOT:
#    python pipeline/run_pipeline.py data/raw/IMG_t1.JPG data/raw/IMG_t2.JPG
#
#  Or with options:
#    python pipeline/run_pipeline.py data/raw/IMG_t1.JPG data/raw/IMG_t2.JPG \
#        --out data/results/experiment_01  \
#        --debug-preprocess
# ─────────────────────────────────────────────

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Make sure pipeline/ is on the import path ──
# Works whether you run from the project root or from inside pipeline/
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Project root = one level up from pipeline/
PROJECT_ROOT = PIPELINE_DIR.parent


def main():
    parser = argparse.ArgumentParser(
        description="Arabidopsis hypocotyl growth measurement pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run — images in data/raw/, results go to data/results/ automatically
  python pipeline/run_pipeline.py data/raw/IMG_1721.JPG data/raw/IMG_1731.JPG

  # Custom output folder
  python pipeline/run_pipeline.py data/raw/t1.JPG data/raw/t2.JPG --out data/results/exp01

  # Check preprocessing only (fast, no segmentation)
  python pipeline/run_pipeline.py data/raw/t1.JPG data/raw/t2.JPG --debug-preprocess

  # Skip alignment (if images are already aligned)
  python pipeline/run_pipeline.py data/raw/t1.JPG data/raw/t2.JPG --no-align
        """
    )
    parser.add_argument("image_t1",
                        help="Path to timepoint 1 image (e.g. data/raw/IMG_1721.JPG)")
    parser.add_argument("image_t2",
                        help="Path to timepoint 2 image (e.g. data/raw/IMG_1731.JPG)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: data/results/<t1>_vs_<t2>/)")
    parser.add_argument("--debug", action="store_true",
                        help="Show intermediate debug plots at each step")
    parser.add_argument("--debug-preprocess", action="store_true",
                        help="Run preprocessing only and show diagnostic plots, then exit")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip image alignment step")
    parser.add_argument("--hypo-fraction", type=float, default=None,
                        help="Override hypocotyl/root split fraction (0–1, default from config.py)")
    args = parser.parse_args()

    # ── Resolve paths ──
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

    # ── Config overrides ──
    import config
    if args.hypo_fraction is not None:
        config.HYPOCOTYL_FRACTION = args.hypo_fraction
        print(f"[config] Hypocotyl fraction overridden to {args.hypo_fraction}")

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 1 – Preprocessing")
    print("="*55)
    from step1_preprocess import preprocess

    dbg = args.debug or args.debug_preprocess
    crop_t1, roi_t1, ppm_t1, div_x_t1 = preprocess(str(t1_path), debug=dbg)
    crop_t2, roi_t2, ppm_t2, div_x_t2 = preprocess(str(t2_path), debug=dbg)

    print(f"\n  t1 crop : {crop_t1.shape}  px/mm: {ppm_t1}  divider x={div_x_t1}")
    print(f"  t2 crop : {crop_t2.shape}  px/mm: {ppm_t2}  divider x={div_x_t2}")

    if args.debug_preprocess:
        print("\n  --debug-preprocess: stopping after Step 1.")
        print("  Adjust config.py if needed, then re-run without this flag.")
        return

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
        div_x = (div_x_t1 + div_x_t2) // 2
        print(f"\n  Alignment skipped. Average divider x={div_x}.")

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 3 – Segmentation")
    print("="*55)
    from step3_segmentation import segment_side

    t1_left  = crop_t1[:, :div_x]
    t1_right = crop_t1[:, div_x:]
    t2_left  = crop_t2_aligned[:, :div_x]
    t2_right = crop_t2_aligned[:, div_x:]

    t0 = time.time()
    print("\n  → t1 Left")
    inst_t1_left  = segment_side(t1_left,  "t1-left")
    print("\n  → t1 Right")
    inst_t1_right = segment_side(t1_right, "t1-right")
    print("\n  → t2 Left")
    inst_t2_left  = segment_side(t2_left,  "t2-left")
    print("\n  → t2 Right")
    inst_t2_right = segment_side(t2_right, "t2-right")
    print(f"\n  Segmentation done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 4 – Cross-timepoint matching")
    print("="*55)
    from step4_matching import match_instances

    print("\n  Left side:")
    matches_left  = match_instances(inst_t1_left,  inst_t2_left)
    print("\n  Right side:")
    matches_right = match_instances(inst_t1_right, inst_t2_right)

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 5 – QC visualization")
    print("="*55)
    from step5_qc_visualization import make_qc_figure

    qc_path = str(qc_dir / f"qc_{t1_name}_vs_{t2_name}.png")
    make_qc_figure(
        t1_left, t1_right, t2_left, t2_right,
        inst_t1_left, inst_t1_right,
        inst_t2_left, inst_t2_right,
        matches_left, matches_right,
        div_x_t1=div_x,
        out_path=qc_path,
    )

    # ══════════════════════════════════════════
    print("\n" + "="*55)
    print("  STEP 6 – Measurements & export")
    print("="*55)
    from step6_export import compile_measurements, export_excel

    df = compile_measurements(
        matches_left, matches_right,
        px_per_mm_t1=ppm_t1 or 1.0,
        px_per_mm_t2=ppm_t2 or 1.0,
    )

    print(f"\n  Measurement table ({len(df)} rows):")
    print(df[['genotype_side', 'label', 'length_t1_mm', 'length_t2_mm',
              'delta_mm', 'growth_rate_pct', 'matched', 'notes']].to_string(index=False))

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
