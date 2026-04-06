# ─────────────────────────────────────────────
#  step3_click_interface.py
#
#  Two-phase annotation:
#    Phase A (t1): click cotyledon TOP then root TIP per seedling
#    Phase B (t2): verify/correct suggested positions (pre-populated
#                  from homography-mapped t1 clicks)
# ─────────────────────────────────────────────

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
#  T1 annotation window
#  Click alternating: cotyledon top, root tip, cotyledon top, root tip...
# ─────────────────────────────────────────────

class T1Collector:
    """
    Alternating clicks:
      odd  click (1, 3, 5...) = cotyledon top  (green dot)
      even click (2, 4, 6...) = root tip / hypocotyl bottom  (red dot)
    Each pair = one seedling.
    """

    def __init__(self, ax, img_shape):
        self.ax        = ax
        self.h, self.w = img_shape[:2]
        self.pairs     = []          # list of {'top': (r,c), 'bot': (r,c)}
        self._pending_top = None     # top click waiting for its bottom
        self._artists  = []          # for undo
        self._confirmed = False
        self._n_clicks  = 0

        self._status = ax.text(
            0.01, 0.015,
            self._step_msg(),
            transform=ax.transAxes,
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='#1a6e1a', alpha=0.92),
            va='bottom', zorder=10,
        )

    def _step_msg(self):
        n = len(self.pairs)
        if self._pending_top is None:
            return (f"Click {n+1}: COTYLEDON TOP  (green leaf above hypocotyl) "
                    f"— {n} seedling(s) done so far")
        else:
            return (f"Click {n+1}: ROOT TIP  (bottom of hypocotyl, where root begins) "
                    f"— then next cotyledon top")

    def on_click(self, event):
        if event.inaxes != self.ax or self._confirmed:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.button == 3:
            self._undo()
            return

        r, c = int(y), int(x)
        self._n_clicks += 1

        if self._pending_top is None:
            # This is a cotyledon top click
            self._pending_top = (r, c)
            n = len(self.pairs) + 1
            color = PALETTE[(n - 1) % len(PALETTE)]
            dot = self.ax.plot(c, r, 'o', color=color,
                               markersize=13, markeredgecolor='white',
                               markeredgewidth=2.5, zorder=9)[0]
            lbl = self.ax.text(c + 12, r - 6, f"{n}▲",
                               color='white', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2',
                                         facecolor=color, alpha=0.88),
                               zorder=10)
            self._artists.append({'kind': 'top', 'color': color,
                                   'artists': [dot, lbl], 'top': (r, c)})
        else:
            # This is the root tip click for the pending top
            top = self._pending_top
            n   = len(self.pairs) + 1
            color = PALETTE[(n - 1) % len(PALETTE)]

            # Draw line from top to bottom + bottom dot
            line = self.ax.plot([top[1], c], [top[0], r],
                                '--', color=color, linewidth=1.5,
                                alpha=0.7, zorder=8)[0]
            dot  = self.ax.plot(c, r, 's', color=color,
                                markersize=11, markeredgecolor='white',
                                markeredgewidth=2, zorder=9)[0]
            lbl  = self.ax.text(c + 12, r + 4, f"{n}▼",
                                color='white', fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2',
                                          facecolor=color, alpha=0.88),
                                zorder=10)
            self._artists.append({'kind': 'bot', 'color': color,
                                   'artists': [dot, lbl, line]})
            self.pairs.append({'top': top, 'bot': (r, c), 'color': color})
            self._pending_top = None

        self._update_status()
        self.ax.figure.canvas.draw_idle()

    def _undo(self):
        if not self._artists:
            return
        entry = self._artists.pop()
        for a in entry['artists']:
            a.remove()
        if entry['kind'] == 'bot':
            self.pairs.pop()
        self._pending_top = None
        self._update_status()
        self.ax.figure.canvas.draw_idle()

    def _update_status(self):
        self._status.set_text(self._step_msg())
        color = '#1a6e1a' if self._pending_top is None else '#8b4500'
        self._status.get_bbox_patch().set_facecolor(color)

    def on_key(self, event):
        if event.key == 'enter':
            if self._pending_top is not None:
                print("  [click] Complete the current seedling "
                      "(click the root tip) before confirming")
                return
            if not self.pairs:
                print("  [click] Click at least one seedling first")
                return
            self._confirmed = True
            self._status.set_text(
                f"✓ {len(self.pairs)} seedlings confirmed — running...")
            self._status.get_bbox_patch().set_facecolor('#1a1ae8')
            self.ax.figure.canvas.draw_idle()
            plt.close(self.ax.figure)


# ─────────────────────────────────────────────
#  T2 verification window
#  Shows suggested positions from t1 mapping.
#  User can move dots or accept as-is.
# ─────────────────────────────────────────────

class T2Collector:
    """
    Pre-populated with suggested (top, bot) pairs from t1 homography mapping.
    User can:
      - Accept all and press Enter
      - Left-click near a dot to drag it (click = move to new position)
      - Right-click a pair number to delete the whole pair
    """

    def __init__(self, ax, img_shape, suggested_pairs):
        self.ax          = ax
        self.h, self.w   = img_shape[:2]
        self.pairs       = [{'top': p['top'], 'bot': p['bot'],
                              'color': p['color']}
                            for p in suggested_pairs]
        self._artists    = []
        self._confirmed  = False
        self._selected   = None   # (pair_idx, 'top'|'bot') being moved

        self._status = ax.text(
            0.01, 0.015,
            "CHECK suggested positions. Left-click to move a point. "
            "Right-click a number to delete pair. Press ENTER to accept.",
            transform=ax.transAxes,
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='#1a1a8b', alpha=0.92),
            va='bottom', zorder=10,
        )
        self._draw_all()

    def _draw_all(self):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists = []

        for i, p in enumerate(self.pairs):
            color = p['color']
            n     = i + 1
            tr, tc = p['top']
            br, bc = p['bot']

            line = self.ax.plot([tc, bc], [tr, br], '--',
                                color=color, linewidth=1.5,
                                alpha=0.6, zorder=8)[0]
            top_dot = self.ax.plot(tc, tr, 'o', color=color,
                                   markersize=13, markeredgecolor='white',
                                   markeredgewidth=2.5, zorder=9)[0]
            bot_dot = self.ax.plot(bc, br, 's', color=color,
                                   markersize=11, markeredgecolor='white',
                                   markeredgewidth=2, zorder=9)[0]
            lbl_t = self.ax.text(tc + 12, tr - 6, f"{n}▲",
                                  color='white', fontsize=10, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2',
                                            facecolor=color, alpha=0.88),
                                  zorder=10)
            lbl_b = self.ax.text(bc + 12, br + 4, f"{n}▼",
                                  color='white', fontsize=10, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2',
                                            facecolor=color, alpha=0.88),
                                  zorder=10)
            self._artists += [line, top_dot, bot_dot, lbl_t, lbl_b]

        self.ax.figure.canvas.draw_idle()

    def _nearest_point(self, r, c, threshold=40):
        """Find the nearest (pair_idx, 'top'|'bot') within threshold px."""
        best_dist = threshold
        best      = None
        for i, p in enumerate(self.pairs):
            for key in ('top', 'bot'):
                pr, pc = p[key]
                d = np.sqrt((r - pr)**2 + (c - pc)**2)
                if d < best_dist:
                    best_dist = d
                    best      = (i, key)
        return best

    def on_click(self, event):
        if event.inaxes != self.ax or self._confirmed:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        r, c = int(y), int(x)

        if event.button == 3:
            # Right-click: delete nearest pair
            hit = self._nearest_point(r, c, threshold=60)
            if hit:
                self.pairs.pop(hit[0])
                self._draw_all()
            return

        if event.button == 1:
            # Left-click: move nearest point
            if self._selected is None:
                hit = self._nearest_point(r, c)
                if hit:
                    self._selected = hit
                    self._status.set_text(
                        f"Moving point {hit[0]+1} {hit[1]} — "
                        f"click new position")
                    self.ax.figure.canvas.draw_idle()
            else:
                idx, key = self._selected
                self.pairs[idx][key] = (r, c)
                self._selected = None
                self._status.set_text(
                    "CHECK suggested positions. Left-click to move a point. "
                    "Right-click to delete. Press ENTER to accept.")
                self._draw_all()

    def on_key(self, event):
        if event.key == 'enter':
            if not self.pairs:
                print("  [click] No pairs — exiting")
                return
            self._confirmed = True
            self._status.set_text(
                f"✓ {len(self.pairs)} pairs accepted — saving...")
            self._status.get_bbox_patch().set_facecolor('#1ae81a')
            self.ax.figure.canvas.draw_idle()
            plt.close(self.ax.figure)


# ─────────────────────────────────────────────
#  Public session functions
# ─────────────────────────────────────────────

def _instruction_panel(fig, lines):
    ax_i = fig.add_axes([0.72, 0.05, 0.27, 0.90])
    ax_i.set_facecolor('#111122')
    ax_i.set_xlim(0, 1); ax_i.set_ylim(0, 1); ax_i.axis('off')
    n = len(lines)
    for j, (txt, col, bold) in enumerate(lines):
        y = 0.97 - j * (0.97 / (n + 1))
        ax_i.text(0.04, y, txt, color=col,
                  fontsize=8.5, fontweight='bold' if bold else 'normal',
                  fontfamily='monospace', va='top',
                  transform=ax_i.transAxes, clip_on=True)


def run_t1_session(img_crop: np.ndarray, title: str) -> list:
    """
    Returns list of dicts: [{'top': (r,c), 'bot': (r,c), 'color': hex}, ...]
    top = cotyledon position, bot = root tip / hypocotyl bottom
    """
    fig = plt.figure(figsize=(18, 10), facecolor='#0d0d1a')
    ax  = fig.add_axes([0.01, 0.05, 0.70, 0.90])
    ax.set_facecolor('#0d0d1a')
    ax.imshow(img_crop)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')

    _instruction_panel(fig, [
        ("T1 ANNOTATION", "#FFD700", True),
        ("", "white", False),
        ("Click PAIRS per seedling:", "#64FFFF", True),
        ("", "white", False),
        ("  1st click = COTYLEDON TOP", "#64FF64", False),
        ("     (green leaf at top of plant)", "white", False),
        ("     → green circle ▲ appears", "#64FF64", False),
        ("", "white", False),
        ("  2nd click = ROOT TIP", "#FF9040", False),
        ("     (where hypocotyl ends,", "white", False),
        ("      root begins)", "white", False),
        ("     → square ▼ + dashed line", "#FF9040", False),
        ("", "white", False),
        ("Repeat for every seedling.", "white", False),
        ("Both left AND right side.", "white", False),
        ("", "white", False),
        ("─" * 35, "#444444", False),
        ("", "white", False),
        ("  Right-click = undo last", "#FFAAAA", False),
        ("  ENTER = confirm all", "#AAFFAA", False),
    ])

    collector = T1Collector(ax, img_crop.shape)
    fig.canvas.mpl_connect('button_press_event', collector.on_click)
    fig.canvas.mpl_connect('key_press_event',    collector.on_key)

    print("\n" + "─"*60)
    print("  T1 ANNOTATION WINDOW")
    print("─"*60)
    print("  For each seedling, click TWO points:")
    print("  1st click → COTYLEDON TOP (green leaf above hypocotyl)")
    print("  2nd click → ROOT TIP (where hypocotyl ends, root begins)")
    print("  Repeat for all seedlings (left + right side)")
    print("  Right-click = undo last | ENTER = confirm")
    print("─"*60)

    plt.show(block=True)

    if not collector._confirmed or not collector.pairs:
        print("  No annotations — exiting.")
        import sys; sys.exit(0)

    print(f"  [click] {len(collector.pairs)} seedlings annotated")
    return collector.pairs


def run_t2_session(img_t2: np.ndarray, title: str,
                   suggested_pairs: list) -> list:
    """
    Show t2 with suggested pairs pre-populated.
    User verifies/corrects. Returns confirmed pairs.
    """
    fig = plt.figure(figsize=(18, 10), facecolor='#0d0d1a')
    ax  = fig.add_axes([0.01, 0.05, 0.70, 0.90])
    ax.set_facecolor('#0d0d1a')
    ax.imshow(img_t2)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')

    _instruction_panel(fig, [
        ("T2 VERIFICATION", "#FFD700", True),
        ("", "white", False),
        ("Suggested positions shown.", "white", False),
        ("Check each seedling:", "#64FFFF", True),
        ("", "white", False),
        ("  ▲ circle = cotyledon top", "#64FF64", False),
        ("  ▼ square = root tip", "#FF9040", False),
        ("", "white", False),
        ("To MOVE a point:", "#FFFF64", True),
        ("  Left-click the dot,", "white", False),
        ("  then left-click new position", "white", False),
        ("", "white", False),
        ("To DELETE a seedling:", "#FFAAAA", True),
        ("  Right-click near its number", "white", False),
        ("", "white", False),
        ("─" * 35, "#444444", False),
        ("", "white", False),
        ("  ENTER = accept all", "#AAFFAA", False),
    ])

    collector = T2Collector(ax, img_t2.shape, suggested_pairs)
    fig.canvas.mpl_connect('button_press_event', collector.on_click)
    fig.canvas.mpl_connect('key_press_event',    collector.on_key)

    print("\n" + "─"*60)
    print("  T2 VERIFICATION WINDOW")
    print("─"*60)
    print("  Suggested positions pre-filled from t1 + alignment.")
    print("  Left-click a dot → left-click new position to move it.")
    print("  Right-click near a number to delete that seedling.")
    print("  ENTER = accept all")
    print("─"*60)

    plt.show(block=True)

    if not collector._confirmed or not collector.pairs:
        print("  No pairs confirmed — exiting.")
        import sys; sys.exit(0)

    print(f"  [click] {len(collector.pairs)} t2 pairs confirmed")
    return collector.pairs


# ─────────────────────────────────────────────
#  QC overlay window
# ─────────────────────────────────────────────

def show_qc_window(img_t1, img_t2_aligned,
                   results_t1, results_t2,
                   px_per_mm_t1, px_per_mm_t2,
                   out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(20, 11),
                             facecolor='#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    for ax, img, results, px_per_mm, label in [
        (axes[0], img_t1,         results_t1, px_per_mm_t1, "t1  (annotated)"),
        (axes[1], img_t2_aligned, results_t2, px_per_mm_t2, "t2  (verified)"),
    ]:
        ax.set_facecolor('#0d0d1a')
        ax.imshow(img)

        for res in results:
            if not res or not res.get('path'):
                continue
            i      = res['label'] - 1
            color  = PALETTE[i % len(PALETTE)]
            path   = res['path']
            rows   = [p[0] for p in path]
            cols   = [p[1] for p in path]
            ax.plot(cols, rows, '-', color=color,
                    linewidth=3, alpha=0.85)
            sr, sc = res['start']
            ax.plot(sc, sr, 'o', color=color, markersize=11,
                    markeredgecolor='white', markeredgewidth=2)
            er, ec = res['end']
            ax.plot(ec, er, 's', color=color, markersize=9,
                    markeredgecolor='white', markeredgewidth=2)
            mm     = res.get('length_mm')
            mm_str = f"{mm:.1f} mm" if mm else f"{res['length_px']:.0f} px"
            mid    = len(path) // 2
            mr, mc = path[mid]
            ax.text(mc + 12, mr, f"{res['label']}: {mm_str}",
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor=color, alpha=0.85))

        ax.set_title(label, color='white', fontsize=13,
                     fontweight='bold', pad=6)
        ax.axis('off')

    plt.suptitle(
        "QC — Check traced paths match the hypocotyls.\n"
        "Close this window or press ENTER to save results.",
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
