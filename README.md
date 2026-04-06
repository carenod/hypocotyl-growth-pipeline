# Arabidopsis Hypocotyl Growth Pipeline

Automated pipeline to measure hypocotyl elongation in *Arabidopsis thaliana* seedlings from pairs of camera images taken at two timepoints.

---

## What it does

1. **Preprocesses** each image: detects the plate boundary, calibrates a px/mm scale from the ruler, and splits the plate into left/right genotype sides
2. **Aligns** the two timepoint images to a common coordinate space using feature matching
3. **Segments** each hypocotyl using PlantCV adaptive thresholding on the LAB colour space
4. **Filters** detections by shape (area, aspect ratio, solidity) and flags tangled hypocotyls
5. **Matches** each hypocotyl in t1 to the same hypocotyl in t2 by position
6. **Exports** a QC overlay image (numbered, colour-coded) and an Excel file with lengths in mm and growth rates

## Directory structure

```
hypocotyl-growth-pipeline/
├── pipeline/               ← all analysis code (tracked by Git)
│   ├── config.py           ← tunable parameters
│   ├── run_pipeline.py     ← main entry point
│   ├── step1_preprocess.py
│   ├── step2_alignment.py
│   ├── step3_segmentation.py
│   ├── step4_matching.py
│   ├── step5_qc_visualization.py
│   └── step6_export.py
├── data/                   ← images and results (git-ignored, stays local)
│   ├── raw/                ← place your image pairs here
│   └── results/            ← pipeline outputs go here automatically
├── docs/                   ← figures and notes
├── tests/                  ← test scripts
├── environment.yml         ← conda environment definition
└── .gitignore
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hypocotyl-growth-pipeline.git
cd hypocotyl-growth-pipeline
```

### 2. Create the conda environment
```bash
conda env create -f environment.yml
conda activate hypocotyl
```

> **GPU note:** the pipeline does not require a GPU but PlantCV runs faster with one.  
> Check your CUDA version with `nvidia-smi` and adjust `pytorch-cuda` in `environment.yml` if needed.

## Usage

### Basic run
Place your image pair in `data/raw/`, then run from the **project root**:

```bash
python pipeline/run_pipeline.py data/raw/IMG_t1.JPG data/raw/IMG_t2.JPG
```

Results are saved automatically to `data/results/IMG_t1_vs_IMG_t2/`.

### Options

| Flag | Description |
|---|---|
| `--out PATH` | Custom output directory |
| `--debug-preprocess` | Check plate crop, ruler, and divider detection only — no segmentation |
| `--no-align` | Skip image alignment |
| `--hypo-fraction 0.6` | Override hypocotyl/root split fraction at runtime |

### Tune the segmentation threshold

Before running the full pipeline on a new image set, tune the threshold:

```bash
python pipeline/step3_segmentation.py data/raw/IMG_t1.JPG
```

This saves a tuning grid to `data/results/tuning/` showing 12 threshold values.  
Pick the one where hypocotyls appear as clean white filaments, then set `PCV_THRESH` in `pipeline/config.py`.

## Output files

```
data/results/IMG_t1_vs_IMG_t2/
├── measurements_IMG_t1_vs_IMG_t2.xlsx   ← main results
└── qc/
    └── qc_IMG_t1_vs_IMG_t2.png          ← visual QC overlay
```

### Excel sheets

| Sheet | Contents |
|---|---|
| All measurements | One row per detected hypocotyl pair |
| Left (A) / Right (B) | Per-genotype tables |
| Summary | Mean ± SE per genotype |
| Calibration | px/mm ratios derived from ruler |

### QC image legend

| Colour / label | Meaning |
|---|---|
| Coloured + number | Accepted, matched between t1 and t2 |
| Red + T | Tangled — excluded |
| Grey + ? | Detected but not matched |
| Cyan line | Hypocotyl/root split point |
| Yellow line | Skeleton used for measurement |

## Key parameters (`pipeline/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `PCV_THRESH` | 100 | Absolute LAB-L threshold for segmentation |
| `HYPOCOTYL_FRACTION` | 0.55 | Fraction of skeleton measured as hypocotyl |
| `MIN_AREA_PX` | 5000 | Minimum blob area to keep |
| `MIN_ASPECT_RATIO` | 2.5 | Minimum elongation ratio |
| `MAX_JUNCTION_POINTS` | 6 | Branch points above this → tangled |
| `MAX_MATCH_DISTANCE_PX` | 200 | Max centroid shift for t1↔t2 matching |

## Working across two computers

```bash
# Before starting work — always pull latest changes
git pull

# After finishing work — commit and push
git add pipeline/
git commit -m "describe what you changed"
git push
```

> Images (`data/`) are git-ignored and stay local on each machine.  
> Copy new image pairs manually (USB, OneDrive, etc.) to `data/raw/` on each computer.

## Citation

If you use this pipeline in your research, please cite:  
> [your name] (*year*). Hypocotyl growth pipeline. GitHub: https://github.com/YOUR_USERNAME/hypocotyl-growth-pipeline
