#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  run_pipeline.py  –  main entry point
#
#  Single-pair mode:
#    python pipeline/run_pipeline.py --onePair data/raw/IMG_t1.JPG data/raw/IMG_t2.JPG
#
#  Batch mode:
#    python pipeline/run_pipeline.py --batch data/experiment_folder/
# ─────────────────────────────────────────────

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

PROJECT_ROOT = PIPELINE_DIR.parent

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_number(path: Path) -> int:
    """Return the last integer found in a filename stem, for sorting by time."""
    nums = re.findall(r'\d+', path.stem)
    return int(nums[-1]) if nums else 0


def _find_images_in_subfolders(folder: Path) -> dict[str, list[Path]]:
    """
    Return {subfolder_name: [sorted images]} for every subfolder that has
    at least 2 images.  Sorted by number in filename (lower = earlier).
    """
    result = {}
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        images = sorted(
            [f for f in sub.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
            key=_extract_number,
        )
        if len(images) < 2:
            print(f"  [batch] Skipping {sub.name}: fewer than 2 images found.")
        else:
            result[sub.name] = images
    return result


def _ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ('y', 'yes'):
            return True
        if ans in ('n', 'no', ''):
            return False


# ── Core pipeline for one t1/t2 pair ─────────────────────────────────────────

def process_pair(t1_path: Path, t2_path: Path,
                 out_dir: Path, qc_dir: Path,
                 no_align: bool = False,
                 debug: bool = False,
                 manual_annotation: bool = True,
                 manual_verification: bool = True,
                 save_qc: bool = True,
                 given_pairs_t1: list = None) -> tuple[pd.DataFrame, list]:
    """
    Run the full pipeline for one (t1, t2) image pair.

    Parameters
    ----------
    manual_annotation   : show the t1 click window (False = use given_pairs_t1)
    manual_verification : show the t2 verification window (False = auto-accept)
    save_qc             : save the QC overlay image
    given_pairs_t1      : pre-defined t1 click positions (used when
                          manual_annotation=False)

    Returns
    -------
    (df, pairs_t2) — measurements DataFrame and the t2 click positions,
    which can be passed as given_pairs_t1 to the next pair in a sequence.
    """
    from step1_preprocess import preprocess
    from step2_alignment import align_images
    from step3_click_interface import run_t1_session, run_t2_session, show_qc_window
    from step3_tracer import trace_all
    import config
    import matplotlib.pyplot as plt

    t1_name = t1_path.stem
    t2_name = t2_path.stem

    print(f"\n  t1 : {t1_path.name}")
    print(f"  t2 : {t2_path.name}")

    # ── Step 1: Preprocessing ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 1 – Preprocessing")
    print("="*55)

    crop_t1, roi_t1, ppm_t1, div_x_t1 = preprocess(str(t1_path), debug=debug)
    crop_t2, roi_t2, ppm_t2, div_x_t2 = preprocess(str(t2_path), debug=debug)

    print(f"\n  t1: {crop_t1.shape}  px/mm={ppm_t1}  divider x={div_x_t1}")
    print(f"  t2: {crop_t2.shape}  px/mm={ppm_t2}  divider x={div_x_t2}")

    div_x = div_x_t1

    # ── Step 2: Alignment ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 2 – Alignment  (t2 → t1 coordinate space)")
    print("="*55)

    if not no_align:
        landmark_path = str(qc_dir / f"alignment_landmarks_{t1_name}_vs_{t2_name}.png")
        crop_t2_aligned, M = align_images(crop_t1, crop_t2,
                                          debug=debug,
                                          save_path=landmark_path)
        print(f"  Alignment landmarks → {landmark_path}")
    else:
        crop_t2_aligned = crop_t2
        M = None
        div_x = (div_x_t1 + div_x_t2) // 2
        print("  Alignment skipped.")

    aligned_path = str(qc_dir / f"aligned_t2_{t2_name}.png")
    plt.figure(figsize=(12, 9))
    plt.imshow(crop_t2_aligned)
    plt.title(f"Aligned t2: {t2_name}\n(check this matches t1 layout)", fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(aligned_path, dpi=120)
    plt.close()
    print(f"  Aligned t2 saved → {aligned_path}")

    # ── Step 3a: Annotate t1 ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  STEP 3a – t1 annotation")
    print("="*55)

    config.PX_PER_MM_HINT = ppm_t1 or 33.0

    if manual_annotation:
        pairs_t1 = run_t1_session(
            crop_t1,
            title=f"t1: {t1_path.name}  —  Click COTYLEDON then ROOT TIP per seedling",
            image_path=str(t1_path),
        )
    else:
        pairs_t1 = given_pairs_t1
        print("  Using positions carried forward from previous timepoint.")

    for i, p in enumerate(pairs_t1):
        r, c = p['top']
        side = "Left (A)" if c < div_x else "Right (B)"
        print(f"  [{i+1}] top=({r},{c}) bot={p['bot']} → {side}")

    print("\n  → Tracing hypocotyls in t1...")
    results_t1 = trace_all(crop_t1, pairs_t1, px_per_mm=ppm_t1)

    # ── Step 3b: Verify / accept t2 positions ────────────────────────────
    print("\n" + "="*55)
    print("  STEP 3b – t2 positions")
    print("="*55)

    suggested = [{'top': p['top'], 'bot': p['bot'], 'color': p['color']}
                 for p in pairs_t1]

    if manual_verification:
        pairs_t2 = run_t2_session(
            crop_t2_aligned,
            title=f"t2: {t2_path.name}  —  Verify / correct suggested positions",
            suggested_pairs=suggested,
            image_path=str(t2_path),
        )
    else:
        pairs_t2 = suggested
        print("  Positions auto-accepted (intermediate timepoint).")

    # After alignment, crop_t2_aligned is in t1's coordinate space → use ppm_t1
    ppm_for_t2 = ppm_t1 if (not no_align) else ppm_t2
    print("\n  → Tracing hypocotyls in t2...")
    config.PX_PER_MM_HINT = ppm_for_t2 or 33.0
    results_t2 = trace_all(crop_t2_aligned, pairs_t2, px_per_mm=ppm_for_t2)

    # ── Step 4: QC overlay ────────────────────────────────────────────────
    if save_qc:
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
        print(f"  QC image → {qc_path}")

    # ── Step 5: Build measurements DataFrame ─────────────────────────────
    rows = []
    for r1, r2 in zip(results_t1, results_t2):
        label   = r1['label']
        col     = r1['click'][1]
        side    = "Left (A)" if col < div_x else "Right (B)"
        l_t1_px = r1['length_px']
        l_t2_px = r2['length_px'] if r2 else None
        l_t1_mm = r1['length_mm']
        l_t2_mm = r2['length_mm'] if r2 else None
        delta   = (l_t2_mm - l_t1_mm) if (l_t1_mm and l_t2_mm) else None
        growth  = (delta / l_t1_mm * 100) if (delta and l_t1_mm) else None

        rows.append({
            'image_t1'        : t1_path.name,
            'image_t2'        : t2_path.name,
            'genotype_side'   : side,
            'label'           : label,
            'length_t1_px'    : round(l_t1_px, 1),
            'length_t2_px'    : round(l_t2_px, 1)  if l_t2_px else None,
            'length_t1_mm'    : round(l_t1_mm, 3)  if l_t1_mm else None,
            'length_t2_mm'    : round(l_t2_mm, 3)  if l_t2_mm else None,
            'delta_mm'        : round(delta, 3)     if delta   else None,
            'growth_rate_pct' : round(growth, 1)    if growth  else None,
            'matched'         : r2 is not None,
            'notes'           : '',
        })

    df = pd.DataFrame(rows)
    print(f"\n  {len(df)} hypocotyls measured  "
          f"({t1_path.name} → {t2_path.name})")

    return df, pairs_t2


# ── Multi-timepoint sequence (3+ images) ─────────────────────────────────────

def process_sequence(images: list[Path], out_dir: Path, qc_dir: Path,
                     no_align: bool = False, debug: bool = False) -> pd.DataFrame:
    """
    Process a sequence of N images as N-1 consecutive pairs.

    Manual annotation  : first image only
    Manual verification: last image only
    QC overlay saved   : first and last pair only
    Intermediate pairs : positions carried forward automatically
    """
    n_pairs = len(images) - 1
    all_dfs = []
    carried_pairs = None   # t2 positions forwarded to next pair's t1

    for i in range(n_pairs):
        t1_path = images[i]
        t2_path = images[i + 1]
        is_first = (i == 0)
        is_last  = (i == n_pairs - 1)

        pair_label = f"pair_{i+1}_of_{n_pairs}"
        pair_qc    = qc_dir / pair_label
        pair_qc.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*55}")
        print(f"  Timepoint pair {i+1}/{n_pairs}: "
              f"{t1_path.name} → {t2_path.name}")
        if not is_first and not is_last:
            print("  (intermediate — positions carried forward, no QC window)")

        df, carried_pairs = process_pair(
            t1_path, t2_path,
            out_dir   = out_dir,
            qc_dir    = pair_qc,
            no_align  = no_align,
            debug     = debug,
            manual_annotation   = is_first,
            manual_verification = is_last,
            save_qc             = is_first or is_last,
            given_pairs_t1      = carried_pairs if not is_first else None,
        )
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ── Mode: single pair ─────────────────────────────────────────────────────────

def run_single_pair(args):
    from step6_export import export_excel

    t1_path = Path(args.onePair[0])
    t2_path = Path(args.onePair[1])
    t1_name = t1_path.stem
    t2_name = t2_path.stem

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = t1_path.resolve().parent / f"{t1_name}_vs_{t2_name}"

    qc_dir = out_dir / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(exist_ok=True)

    print(f"\n  Image t1 : {t1_path}")
    print(f"  Image t2 : {t2_path}")
    print(f"  Output   : {out_dir}")

    df, _ = process_pair(t1_path, t2_path, out_dir, qc_dir,
                         no_align=args.no_align, debug=args.debug)

    print(f"\n  Measurements ({len(df)} hypocotyls):")
    print(df[['genotype_side', 'label', 'length_t1_mm',
              'length_t2_mm', 'delta_mm', 'growth_rate_pct']].to_string(index=False))

    xlsx_path = out_dir / f"measurements_{t1_name}_vs_{t2_name}.xlsx"
    export_excel(df, str(xlsx_path),
                 px_per_mm_t1=None, px_per_mm_t2=None,
                 image_t1=str(t1_path), image_t2=str(t2_path))

    print("\n" + "="*55)
    print("  DONE")
    print("="*55)
    print(f"  Excel → {xlsx_path}\n")


# ── Mode: batch ───────────────────────────────────────────────────────────────

def run_batch(args):
    from step6_export import export_excel

    exp_folder = Path(args.batch).resolve()
    if not exp_folder.is_dir():
        print(f"ERROR: {exp_folder} is not a directory.")
        sys.exit(1)

    out_dir  = exp_folder / "results"
    qc_base  = out_dir / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_base.mkdir(exist_ok=True)

    subfolders = _find_images_in_subfolders(exp_folder)
    if not subfolders:
        print(f"No image pairs found in {exp_folder}")
        sys.exit(1)

    print(f"\n  Experiment : {exp_folder}")
    print(f"  Found {len(subfolders)} subfolder(s) with images")
    print(f"  Results    : {out_dir}\n")

    all_dfs = []
    for i, (subfolder_name, images) in enumerate(subfolders.items(), 1):
        n = len(images)
        print("─" * 55)
        print(f"  [{i}/{len(subfolders)}]  Subfolder: {subfolder_name}  "
              f"({n} image{'s' if n > 1 else ''})")
        for img in images:
            print(f"    • {img.name}")

        if not _ask_yes_no("  Analyse this subfolder?"):
            print("  Skipped.\n")
            continue

        qc_dir = qc_base / subfolder_name
        qc_dir.mkdir(exist_ok=True)

        if n == 2:
            df, _ = process_pair(
                images[0], images[1],
                out_dir  = out_dir,
                qc_dir   = qc_dir,
                no_align = args.no_align,
                debug    = args.debug,
            )
        else:
            print(f"\n  {n} timepoints → processing {n-1} consecutive pairs.")
            print("  Manual annotation: first image only.")
            print("  Manual verification: last image only.")
            print("  QC saved for first and last pair.\n")
            df = process_sequence(
                images,
                out_dir  = out_dir,
                qc_dir   = qc_dir,
                no_align = args.no_align,
                debug    = args.debug,
            )

        all_dfs.append(df)

    if not all_dfs:
        print("\n  No subfolders were analysed.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n  Combined measurements ({len(combined)} hypocotyls):")
    print(combined[['image_t1', 'image_t2', 'genotype_side', 'label',
                    'length_t1_mm', 'length_t2_mm',
                    'delta_mm', 'growth_rate_pct']].to_string(index=False))

    xlsx_path = out_dir / f"measurements_{exp_folder.name}.xlsx"
    export_excel(combined, str(xlsx_path),
                 px_per_mm_t1=None, px_per_mm_t2=None,
                 image_t1="(batch)", image_t2="(batch)")

    print("\n" + "="*55)
    print("  BATCH DONE")
    print("="*55)
    print(f"  {len(combined)} hypocotyls measured across "
          f"{len(all_dfs)} subfolder(s)")
    print(f"  Excel → {xlsx_path}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Arabidopsis hypocotyl growth measurement pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Single pair:\n"
            "    python pipeline/run_pipeline.py "
            "--onePair data/raw/IMG_t1.JPG data/raw/IMG_t2.JPG\n\n"
            "  Batch (one Excel for the whole experiment):\n"
            "    python pipeline/run_pipeline.py --batch data/my_experiment/\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--onePair", nargs=2, metavar=("IMAGE_T1", "IMAGE_T2"),
                      help="Single pair mode: path to t1 image then t2 image")
    mode.add_argument("--batch", metavar="EXPERIMENT_FOLDER",
                      help="Batch mode: experiment folder whose subfolders "
                           "each contain a sequence of images")

    parser.add_argument("--out",      default=None,
                        help="Output directory (single-pair mode only; "
                             "default: next to the images)")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip image alignment step")
    parser.add_argument("--debug",    action="store_true",
                        help="Show extra debug plots")

    args = parser.parse_args()

    if args.onePair:
        run_single_pair(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
