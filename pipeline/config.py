# ─────────────────────────────────────────────
#  config.py  –  pipeline-wide settings
# ─────────────────────────────────────────────

# ── Threshold ─────────────────────────────────
# Used to create a permissive binary mask of ALL bright structures.
# The seeded flood-fill then selects only structures reachable from
# a cotyledon — so this can be low (capturing hypocotyls) without
# worrying about also capturing marks (they get ignored anyway).
# Run the tuning script to see the histogram and pick a good value.
#   At p75≈93: most hypocotyls visible but much noise
#   At p90≈107: cleaner but some hypocotyls start to disappear
# Recommended starting point: p75 to p85
PCV_THRESH       = 93          # absolute LAB-L threshold (0–255)
PCV_FILL_SIZE    = 300         # remove noise blobs smaller than this (px²)
PCV_CLOSE_KSIZE  = 5           # kernel size to close small gaps in filaments
PCV_PRUNE_SIZE   = 20          # prune skeleton barbs shorter than this (px)

# ── Cotyledon detection ───────────────────────
# Cotyledons are detected as green blobs in the LAB-A channel.
# Each blob becomes a seed for the flood-fill.
#
# COTYLEDON_GREEN_THRESHOLD  : how far below LAB-A neutral (128) a pixel
#   must be to count as green. Lower if cotyledons are missed.
# COTYLEDON_MIN_GREEN_PIXELS : min total green px to use seeded mode.
#   Below this → fallback (shape-based) mode.
# COTYLEDON_MIN_BLOB_PX      : min size of one cotyledon blob.
#   Prevents noise pixels from becoming spurious seeds.
COTYLEDON_GREEN_THRESHOLD  = 10    # LAB-A units below neutral
COTYLEDON_MIN_GREEN_PIXELS = 300   # min green px to activate seeded mode
COTYLEDON_MIN_BLOB_PX      = 80    # min px per cotyledon blob

# ── Seeded flood-fill ─────────────────────────
# From each cotyledon centroid, we flood-fill through the binary mask.
#
# SEED_RADIUS         : radius of the seed disk placed at each cotyledon
#   centroid (px). Larger = more robust connection to the binary mask.
# SEED_UPWARD_DILATION: the binary mask is dilated upward by this many px
#   before flood-fill. This bridges the small gap between a cotyledon
#   (which sits just above the hypocotyl top) and the hypocotyl mask.
#   Raise if seeds are not connecting to hypocotyls.
SEED_RADIUS          = 20     # px — seed disk radius at cotyledon centroid
SEED_UPWARD_DILATION = 40     # px — upward dilation to bridge cotyledon gap

# ── Fallback (etiolated — no green cotyledons) ─
# When no green is detected, use standard connected-component segmentation
# filtered by shape + top-bulge test (hypocotyls widen at the top).
BULGE_TOP_RATIO_MIN   = 1.2   # top/middle width ratio threshold
BULGE_UPPER_RATIO_MIN = 1.1   # upper-quarter/middle width ratio threshold

# ── Instance filters ──────────────────────────
MIN_AREA_PX        = 5_000    # discard tiny instances (px²)
MAX_AREA_PX        = 600_000  # discard huge merged instances (px²)
MIN_ASPECT_RATIO   = 2.5      # hypocotyls are elongated
MIN_SOLIDITY       = 0.15     # allow curved/wavy shapes

# ── Tangle detection ──────────────────────────
MAX_JUNCTION_POINTS = 8       # skeleton branch points above this → tangled

# ── Hypocotyl / root split ────────────────────
HYPOCOTYL_FRACTION = 0.55     # fraction of skeleton (from top) = hypocotyl

# ── Matching across timepoints ────────────────
MAX_MATCH_DISTANCE_PX = 200   # max centroid distance (px) for t1↔t2 matching

# ── Ruler detection ───────────────────────────
RULER_STRIP_FRACTION = 0.12   # fraction of image width occupied by ruler
RULER_TICK_MM        = 1.0    # ruler tick spacing in mm

# ── Output ────────────────────────────────────
QC_ALPHA           = 0.45     # mask overlay transparency
COLORS_PER_SIDE    = 20       # max distinct colors for instance coloring

# ── Tracer (Frangi + Dijkstra) ────────────────
# TRACER_COLUMN_WIDTH_MM : horizontal search width on each side of the
#   click point (mm). The path cannot stray further than this from the
#   click column. ~3mm is a good starting point.
# PX_PER_MM_HINT         : set automatically from ruler detection.
#   Used by the Frangi filter to set scale range.
TRACER_COLUMN_WIDTH_MM = 3.0    # mm either side of click column
PX_PER_MM_HINT         = 33.0   # updated automatically at runtime
