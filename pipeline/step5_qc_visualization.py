# ─────────────────────────────────────────────
#  step5_qc_visualization.py
#  • Overlay colored instance masks on images
#  • Number each hypocotyl with its match label
#  • Mark tangled instances
#  • Show split point (hypocotyl / root boundary)
#  • Save QC images to output folder
# ─────────────────────────────────────────────

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
from pathlib import Path
import config


def _instance_colors(n: int) -> list:
    """Generate n visually distinct RGB colors (0-255)."""
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360 / 360.0   # golden angle
        rgb = hsv_to_rgb([hue, 0.75, 0.95])
        colors.append((int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
    return colors


def draw_overlay(img: np.ndarray, instances: list, matches: list,
                 timepoint: str = "t1", side_offset_x: int = 0) -> np.ndarray:
    """
    Draw colored masks + labels on img.
    instances  : all instances for this side/timepoint
    matches    : match list from step4 (used to get the label number)
    timepoint  : 't1' or 't2'
    side_offset_x : pixel offset if this is the right side crop
    Returns annotated RGB image (same size as img).
    """
    overlay = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    colors = _instance_colors(40)

    # Build lookup: instance id → (match_label, color, matched)
    id_to_label = {}
    for m in matches:
        inst = m[timepoint]
        if inst is not None:
            color_idx = (m['label'] - 1) % len(colors)
            id_to_label[inst['id']] = {
                'label'  : m['label'],
                'color'  : colors[color_idx],
                'matched': m['matched'],
            }

    for inst in instances:
        inst_id = inst['id']
        tangled  = inst['tangled']

        # Color selection
        if tangled:
            color = (220, 50, 50)    # red for tangled
        elif inst_id in id_to_label:
            color = id_to_label[inst_id]['color']
        else:
            color = (180, 180, 180)  # gray for unmatched

        # Draw mask
        mask = inst['mask']
        alpha = config.QC_ALPHA
        for c in range(3):
            overlay[:, :, c][mask] = (
                overlay[:, :, c][mask] * (1 - alpha) + color[c] * alpha
            )

        # Draw skeleton
        skel_color = (255, 255, 0) if not tangled else (255, 100, 100)
        skel_pts = np.column_stack(np.where(inst['skeleton']))
        for r, c_coord in skel_pts:
            if 0 <= r < h and 0 <= c_coord < w:
                cv2.circle(overlay.astype(np.uint8), (c_coord, r), 1, skel_color, -1)

        # Draw split line (hypocotyl/root boundary) – only for non-tangled
        if not tangled and 'split_row' in inst:
            sr = inst['split_row']
            cols_in_mask = np.where(mask[sr, :])[0] if sr < h else []
            if len(cols_in_mask) > 0:
                c0, c1 = cols_in_mask[0], cols_in_mask[-1]
                cv2.line(overlay.astype(np.uint8),
                         (c0, sr), (c1, sr), (0, 220, 255), 2)

        # Draw centroid label
        row, col = inst['centroid']
        row, col = int(row), int(col)
        if inst_id in id_to_label:
            label_str = str(id_to_label[inst_id]['label'])
        elif tangled:
            label_str = "T"
        else:
            label_str = "?"

        cv2.putText(overlay.astype(np.uint8),
                    label_str,
                    (col, row),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_qc_figure(img_t1_left, img_t1_right,
                   img_t2_left, img_t2_right,
                   inst_t1_left, inst_t1_right,
                   inst_t2_left, inst_t2_right,
                   matches_left, matches_right,
                   div_x_t1: int,
                   out_path: str = None):
    """
    Compose a 2×2 QC figure:
      top-left: t1 left side   top-right: t1 right side
      bot-left: t2 left side   bot-right: t2 right side
    """
    panels = [
        (img_t1_left,  inst_t1_left,  matches_left,  "t1", "t1 – Left (genotype A)"),
        (img_t1_right, inst_t1_right, matches_right, "t1", "t1 – Right (genotype B)"),
        (img_t2_left,  inst_t2_left,  matches_left,  "t2", "t2 – Left (genotype A)"),
        (img_t2_right, inst_t2_right, matches_right, "t2", "t2 – Right (genotype B)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for ax, (img, instances, matches, tp, title) in zip(axes, panels):
        ann = draw_overlay(img, instances, matches, timepoint=tp)
        ax.imshow(ann)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=(220/255, 50/255, 50/255), label='Tangled (excluded)'),
        mpatches.Patch(facecolor=(180/255, 180/255, 180/255), label='Unmatched'),
        plt.Line2D([0], [0], color='yellow', linewidth=1.5, label='Skeleton'),
        plt.Line2D([0], [0], color='cyan', linewidth=2, label='Hypo/root split'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle("Hypocotyl Detection QC – Check numbered instances before accepting",
                 fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  [QC] Saved → {out_path}")
    plt.show()
    return fig


def make_simple_qc(img, instances, matches, timepoint, title, out_path=None):
    """Simpler single-image QC panel."""
    ann = draw_overlay(img, instances, matches, timepoint=timepoint)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(ann)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  [QC] Saved → {out_path}")
    plt.show()
    return fig
