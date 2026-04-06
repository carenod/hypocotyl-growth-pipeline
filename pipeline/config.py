# ─────────────────────────────────────────────
#  config.py  –  pipeline-wide settings
# ─────────────────────────────────────────────

# ── PlantCV segmentation ──────────────────────
# Run:  python step3_segmentation.py <your_image.jpg>
# This prints the LAB-L histogram percentiles and saves a tuning grid
# showing 12 different threshold values. Pick the one where hypocotyls
# appear as clean white filaments on a black background.
#
# The tuning script will print something like:
#   LAB-L  p50=72  p75=79  p90=105  p95=130
# Start with a value between p75 and p95.

PCV_THRESH       = 100         # absolute LAB-L threshold (0–255); tune with tuning script
PCV_FILL_SIZE    = 500         # remove blobs smaller than this (px²) before/after closing
PCV_CLOSE_KSIZE  = 7           # closing kernel size to bridge gaps within filaments (odd)
PCV_PRUNE_SIZE   = 20          # prune skeleton barbs shorter than this (px)

# ── Segmentation filters ──────────────────────
# At ~33px/mm: hypocotyls are ~80–130px wide, ~330–990px long → area ~20k–130k px²
# At ~20px/mm: hypocotyls are ~40–80px wide,  ~200–600px long → area ~8k–48k px²
MIN_AREA_PX        = 5_000     # discard tiny blobs (noise, seeds)
MAX_AREA_PX        = 500_000   # discard huge merged blobs
MIN_ASPECT_RATIO   = 2.5       # hypocotyls are elongated; lower if bent ones are missed
MIN_SOLIDITY       = 0.15      # allow curved/wavy shapes

# ── Tangle detection ──────────────────────────
MAX_JUNCTION_POINTS = 6        # skeleton branch points above this → tangled → excluded

# ── Hypocotyl / root split ────────────────────
HYPOCOTYL_FRACTION = 0.55      # fraction of skeleton (from top) counted as hypocotyl

# ── Matching across timepoints ────────────────
MAX_MATCH_DISTANCE_PX = 200    # max centroid distance (px) for t1↔t2 matching

# ── Ruler detection ───────────────────────────
RULER_STRIP_FRACTION = 0.12    # fraction of image width occupied by ruler on the right
RULER_TICK_MM        = 1.0     # ruler tick spacing in mm

# ── Output ────────────────────────────────────
QC_ALPHA           = 0.45      # mask overlay transparency
COLORS_PER_SIDE    = 20        # max distinct colors for instance coloring
