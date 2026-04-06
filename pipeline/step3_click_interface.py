# ─────────────────────────────────────────────
#  step3_click_interface.py
#  Interactive matplotlib window for hypocotyl annotation.
# ─────────────────────────────────────────────

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import cv2
import sys

for backend in ['TkAgg', 'Qt5Agg', 'WXAgg']:
    try:
        matplotlib.use(backend)
        break
    except Exception:
        continue

PALETTE = [
    '#FF6464', '#64FF64', '#6464FF', '#FFFF64',
    '#FF64FF', '#64FFFF', '#FFA040', '#40FFA0',
    '#A040FF', '#FF4080', '#40FF80', '#8040FF',
]


# ─────────────────────────────────────────────
#  Instruction panel (drawn on the right side)
# ─────────────────────────────────────────────

INSTRUCTIONS = [
    ("STEP 1  —  Set the bottom boundary", "#1a73e8", True),
    ("", "#1a73e8", False),
    ("The boundary line separates the hypocotyl", "white", False),
    ("(upper part) from the root (lower part).", "white", False),
    ("", "white", False),
    ("→ Click anywhere on the LEFT side of", "#64FFFF", False),
    ("   the image at the height where roots", "#64FFFF", False),
    ("   begin (roughly where hypocotyls end).", "#64FFFF", False),
    ("→ Then click the same height on the", "#64FFFF", False),
    ("   RIGHT side of the image.", "#64FFFF", False),
    ("→ A cyan dashed line will appear.", "#64FFFF", False),
    ("", "white", False),
    ("─" * 38, "#555555", False),
    ("", "white", False),
    ("STEP 2  —  Click each cotyledon", "#e8891a", True),
    ("", "#e8891a", False),
    ("Cotyledons are the small green/yellow", "white", False),
    ("leaves at the top of each seedling.", "white", False),
    ("", "white", False),
    ("→ Click once on each cotyledon.", "#FFFF64", False),
    ("   A numbered dot appears.", "#FFFF64", False),
    ("→ Click ALL plants (left + right side).", "#FFFF64", False),
    ("→ Right-click a dot to remove it.", "#FFFF64", False),
    ("", "white", False),
    ("─" * 38, "#555555", False),
    ("", "white", False),
    ("STEP 3  —  Confirm", "#1ae85a", True),
    ("", "#1ae85a", False),
    ("→ Press ENTER when done.", "#64FF64", False),
    ("   The pipeline traces each hypocotyl", "#64FF64", False),
    ("   downward to the boundary line.", "#64FF64", False),
]


def _draw_instruction_panel(fig):
    """Draw a fixed instruction panel on the right side of the figure."""
    ax_inst = fig.add_axes([0.72, 0.05, 0.27, 0.90])
    ax_inst.set_facecolor('#1a1a2e')
    ax_inst.set_xlim(0, 1)
    ax_inst.set_ylim(0, 1)
    ax_inst.axis('off')

    n = len(INSTRUCTIONS)
    y_start = 0.97
    line_h  = y_start / (n + 1)

    for i, (text, color, bold) in enumerate(INSTRUCTIONS):
        y = y_start - i * line_h
        ax_inst.text(
            0.04, y, text,
            color=color,
            fontsize=8.5,
            fontweight='bold' if bold else 'normal',
            fontfamily='monospace',
            va='top',
            transform=ax_inst.transAxes,
            clip_on=True,
        )

    # Border
    for spine in ax_inst.spines.values():
        spine.set_edgecolor('#444466')
        spine.set_linewidth(1.5)

    return ax_inst


# ─────────────────────────────────────────────
#  Status bar
# ─────────────────────────────────────────────

STATUS_MSGS = {
    (0, 0): ("STEP 1 of 3:  Click the LEFT side of the image at the height "
             "where hypocotyls end and roots begin",
             "#1a73e8"),
    (0, 1): ("STEP 1 of 3:  Good! Now click the RIGHT side at the same height",
             "#1a73e8"),
    (1, 0): ("STEP 2 of 3:  Click each cotyledon (green/yellow leaf at top "
             "of each seedling) — left AND right side",
             "#e8891a"),
    'n'   : ("STEP 2 of 3:  {n} cotyledon(s) marked  |  "
             "Right-click to undo  |  Press ENTER when all are marked",
             "#e8891a"),
    'done': ("✓ Confirmed — running pipeline...", "#1ae85a"),
}


# ─────────────────────────────────────────────
#  Click collector
# ─────────────────────────────────────────────

class ClickCollector:
    def __init__(self, ax, img_shape):
        self.ax           = ax
        self.h, self.w    = img_shape[:2]
        self.phase        = 0
        self.boundary_pts = []
        self.boundary_row = None
        self.clicks       = []
        self.dot_artists  = []
        self._bline       = None
        self._confirmed   = False

        # Status bar at bottom of image axes
        self._status = ax.text(
            0.01, 0.015,
            STATUS_MSGS[(0, 0)][0],
            transform=ax.transAxes,
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor=STATUS_MSGS[(0, 0)][1], alpha=0.92),
            va='bottom', zorder=10,
        )

    # ── event handlers ──────────────────────

    def on_click(self, event):
        if event.inaxes != self.ax or self._confirmed:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.button == 3:          # right-click → undo last cotyledon
            self._undo()
            return

        if event.button == 1:          # left-click
            if self.phase == 0:
                self._add_boundary_point(x, y)
            else:
                self._add_cotyledon(x, y)

        self.ax.figure.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'enter':
            if self.phase == 0:
                self._print_hint("Draw the boundary line first (2 clicks needed)")
                return
            if len(self.clicks) == 0:
                self._print_hint("Click at least one cotyledon first")
                return
            self._confirmed = True
            self._set_status('done')
            self.ax.figure.canvas.draw_idle()
            plt.close(self.ax.figure)

    # ── boundary line ────────────────────────

    def _add_boundary_point(self, x, y):
        self.boundary_pts.append((x, y))
        n = len(self.boundary_pts)

        # Draw a small cross at the click
        cross = self.ax.plot(x, y, 'c+', markersize=16,
                             markeredgewidth=3, zorder=9)[0]
        self.dot_artists.append({'type': 'boundary', 'artists': [cross]})

        if n == 2:
            # Average the two y values → horizontal boundary row
            self.boundary_row = int((self.boundary_pts[0][1] +
                                     self.boundary_pts[1][1]) / 2)
            if self._bline:
                self._bline.remove()
            self._bline = self.ax.axhline(
                self.boundary_row,
                color='cyan', linewidth=2.5, linestyle='--',
                label='Boundary (hypo/root split)', zorder=8
            )
            # Add label on the line
            self.ax.text(
                10, self.boundary_row - 8,
                '← hypocotyl above  |  root below →',
                color='cyan', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='#000033', alpha=0.7),
                zorder=9,
            )
            self.phase = 1
            self._set_status((1, 0))
        else:
            self._set_status((0, n))

    # ── cotyledon clicks ─────────────────────

    def _add_cotyledon(self, x, y):
        row, col  = int(y), int(x)
        n         = len(self.clicks) + 1
        color     = PALETTE[(n - 1) % len(PALETTE)]

        dot = self.ax.plot(col, row, 'o',
                           color=color, markersize=13,
                           markeredgecolor='white',
                           markeredgewidth=2.5, zorder=9)[0]
        lbl = self.ax.text(
            col + 14, row - 6, str(n),
            color='white', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor=color, alpha=0.9),
            zorder=10,
        )
        # Small downward arrow to hint "trace goes down from here"
        arr = self.ax.annotate(
            '', xy=(col, row + 30), xytext=(col, row + 8),
            arrowprops=dict(arrowstyle='->', color=color,
                            lw=2.0), zorder=9,
        )
        self.clicks.append((row, col))
        self.dot_artists.append({'type': 'cotyledon',
                                  'artists': [dot, lbl, arr]})
        self._set_status('n')

    # ── undo ────────────────────────────────

    def _undo(self):
        # Only undo cotyledon clicks, not boundary points
        cot_entries = [d for d in self.dot_artists if d['type'] == 'cotyledon']
        if not cot_entries:
            self._print_hint("Nothing to undo")
            return
        entry = cot_entries[-1]
        for a in entry['artists']:
            a.remove()
        self.dot_artists.remove(entry)
        self.clicks.pop()
        self._set_status('n')
        self.ax.figure.canvas.draw_idle()

    # ── helpers ──────────────────────────────

    def _set_status(self, key):
        if key == 'n':
            msg, color = STATUS_MSGS['n']
            msg = msg.format(n=len(self.clicks))
        elif key == 'done':
            msg, color = STATUS_MSGS['done']
        else:
            msg, color = STATUS_MSGS[key]
        self._status.set_text(msg)
        self._status.get_bbox_patch().set_facecolor(color)

    def _print_hint(self, text):
        print(f"  [click] Hint: {text}")


# ─────────────────────────────────────────────
#  Main click session
# ─────────────────────────────────────────────

def run_click_session(img_crop: np.ndarray,
                      title: str = "Hypocotyl Annotator") -> tuple:
    """
    Open interactive window. Returns (clicks, boundary_row).
    clicks       : list of (row, col) in img_crop coordinates
    boundary_row : int
    """
    # Leave room on the right for the instruction panel
    fig = plt.figure(figsize=(18, 10), facecolor='#0d0d1a')
    ax  = fig.add_axes([0.01, 0.05, 0.70, 0.90])
    ax.set_facecolor('#0d0d1a')
    ax.imshow(img_crop)
    ax.set_title(title, color='white', fontsize=12,
                 fontweight='bold', pad=8)
    ax.axis('off')

    _draw_instruction_panel(fig)

    collector = ClickCollector(ax, img_crop.shape)
    fig.canvas.mpl_connect('button_press_event', collector.on_click)
    fig.canvas.mpl_connect('key_press_event',    collector.on_key)

    # Print concise terminal instructions too
    print("\n" + "─"*60)
    print("  ANNOTATION WINDOW OPEN")
    print("─"*60)
    print("  STEP 1: Set bottom boundary line")
    print("    • Click on the LEFT side of the image at the height")
    print("      where the hypocotyl ends and the root begins")
    print("      (roughly where the plants stop being thick and")
    print("       become thin transparent roots)")
    print("    • Then click the SAME HEIGHT on the RIGHT side")
    print("    • A cyan dashed line will appear across the image")
    print()
    print("  STEP 2: Click each cotyledon")
    print("    • Cotyledons = the small green/yellow leaves at the")
    print("      TOP of each seedling, just above the hypocotyl")
    print("    • Click once per plant — left AND right side of plate")
    print("    • A numbered coloured dot appears on each click")
    print("    • Right-click to undo the last dot")
    print()
    print("  STEP 3: Press ENTER to confirm and run tracing")
    print("─"*60)

    plt.show(block=True)

    if not collector._confirmed:
        print("  [click] Window closed without confirming — exiting.")
        sys.exit(0)

    print(f"\n  [click] Confirmed: {len(collector.clicks)} cotyledons, "
          f"boundary at row {collector.boundary_row}")
    return collector.clicks, collector.boundary_row


# ─────────────────────────────────────────────
#  QC overlay window
# ─────────────────────────────────────────────

def show_qc_window(img_t1, img_t2_aligned,
                   results_t1, results_t2,
                   boundary_row,
                   px_per_mm_t1, px_per_mm_t2,
                   out_path=None):
    """
    Show side-by-side QC with traced paths on t1 and t2.
    Press Enter or close window to proceed to export.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 11),
                             facecolor='#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    for ax, img, results, px_per_mm, label in [
        (axes[0], img_t1,         results_t1, px_per_mm_t1, "t1  (annotated)"),
        (axes[1], img_t2_aligned, results_t2, px_per_mm_t2, "t2  (auto-traced)"),
    ]:
        ax.set_facecolor('#0d0d1a')
        overlay = img.copy()
        cv2.line(overlay,
                 (0, boundary_row), (overlay.shape[1], boundary_row),
                 (0, 220, 220), 3)
        ax.imshow(overlay)

        for res in results:
            if not res or not res.get('path'):
                continue
            i     = res['label'] - 1
            color = PALETTE[i % len(PALETTE)]
            path  = res['path']
            rows  = [p[0] for p in path]
            cols  = [p[1] for p in path]

            ax.plot(cols, rows, '-', color=color,
                    linewidth=3, alpha=0.85)

            sr, sc = res['start']
            ax.plot(sc, sr, 'o', color=color, markersize=11,
                    markeredgecolor='white', markeredgewidth=2)

            mm     = res.get('length_mm')
            mm_str = f"{mm:.1f} mm" if mm else f"{res['length_px']:.0f} px"
            mid    = len(path) // 2
            mr, mc = path[mid]
            ax.text(mc + 12, mr,
                    f"{res['label']}: {mm_str}",
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor=color, alpha=0.85))

        ax.set_title(label, color='white', fontsize=13,
                     fontweight='bold', pad=6)
        ax.axis('off')

    plt.suptitle(
        "QC — Check traced paths match the hypocotyls.\n"
        "Close this window or press ENTER to save results to Excel.",
        color='white', fontsize=11, y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  [QC] Saved → {out_path}")

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)
