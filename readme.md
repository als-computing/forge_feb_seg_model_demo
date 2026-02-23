# Multi-Model Segmentation Pipeline

A comprehensive segmentation pipeline combining SAM3, Cellpose, and DINO models for biological image analysis.

## Overview

This pipeline performs semantic and instance segmentation on microscopy images using three state-of-the-art models:
- **SAM3**: Segment Anything Model 3 for text-prompted segmentation
- **DINO**: Vision transformer for semantic segmentation

The pipeline includes post-processing steps that combine instance and semantic segmentation results to improve accuracy and reduce false positives.

## Environment Setup

Due to package version conflicts, we need to create **two separate conda environments**:

### Environment 1: SAM3
```bash
conda create -n sam3_demo python==3.12
conda activate sam3_demo
pip install .
```

### Environment 2: DINO and Cellpose
```bash
conda create -n dino_demo python==3.12
conda activate dino_demo
pip install -r requirements.txt
```

## Quick Start

### Run Full Pipeline (Single Node)

To run the complete pipeline on NERSC:

```bash
./run_single_node_all_7classes.sh
```

This script will:
1. Run SAM3 inference (uses `sam3_demo` environment)
2. Run DINO inference (uses `dino_demo` environment)
3. Run post-processing (CPU-based, `dino_demo` environment):
   - Filter and combine SAM3 + DINO results
   - Optionally extract Hydrated and Dehydrated Xylem regions

> **Note**: SAM3 and DINO inference can be parallelized in production deployment. Post-processing steps are CPU-based and independent of each other.

### Run with Multiple Nodes (Slurm)

```bash
chmod +x submit_nersc_v6_reserved.sh
sbatch submit_nersc_v6_reserved.sh
squeue -u $USER
```

## Pipeline Architecture

### Stage 1: Model Inference (Parallel)
- **SAM3**: Text-prompted segmentation for multiple anatomical classes
- **DINO**: Semantic segmentation for tissue classification

### Stage 2: Post-Processing (CPU-based, Parallel)
- **SAM3 + DINO** (`combine_sam_dino_v3`): Filters SAM3 predictions using DINO semantic mask guidance; non-xylem classes can optionally be read directly from DINO (`--dino-trust`)
- **Region Extraction**: Crops original images and filters instance masks to semantic regions; supports separate extraction of Hydrated and Dehydrated Xylem vessels

> **Note**: All three post-processing steps are independent and can be run in parallel.

## Pipeline Script Reference

The main pipeline script (`run_single_node_all_7classes.sh`) uses the following key parameters:

### SAM3 Inference
```bash
torchrun --nproc_per_node=4 --nnodes=1 -m src.inference_v6 \
    --patch-size 640 \
    --batch-size 1 \
    --confidence 0.1 \
    --overlap-ratio 0.5 \
    --prompts "Cortex" "Phloem Fibers" "Phloem" "Hydrated Xylem vessels" \
              "Air-based Pith cells" "Water-based Pith cells" "Dehydrated Xylem vessels"
```

### DINO Inference
```bash
torchrun --nproc_per_node=4 --nnodes=1 -m src.inference_dino_v1 \
    --save-overlay
```

### SAM3 + DINO Combination
```bash
python -m src.combine_sam_dino_v3 \
    --dilate-px 5 \
    --save-extracted \
    --dino-trust Cortex Phloem_Fibers Phloem Air-based_Pith_cells Water-based_Pith_cells
```

`--dino-trust` specifies classes to read directly from DINO's semantic mask rather than filtering SAM3 predictions. Classes not in this list (e.g. Xylem) use SAM3 masks filtered by DINO.

## Output Structure

```
results/
│
├── 0220_7classes_envtest_YYYYMMDD_HHMMSS/
│   │
│   ├── sam3/
│   │   ├── Cortex/
│   │   ├── Phloem/
│   │   ├── Phloem_Fibers/
│   │   ├── Air-based_Pith_cells/
│   │   ├── Water-based_Pith_cells/
│   │   ├── Hydrated_Xylem_vessels/
│   │   ├── Dehydrated_Xylem_vessels/
│   │   │   └── {stem}_mask.tif         # Per-class binary masks
│   │   │
│   │   ├── combined/
│   │   │   └── {stem}_combined.tif     # Multi-class combined mask (class IDs 1–7)
│   │   │
│   │   └── combined_overlay/
│   │       └── {stem}_overlay.tif      # Color-coded visualization overlay
│   │
│   ├── dino/
│   │   ├── semantic_masks/
│   │   │   └── {stem}.tif              # Pixel-wise class predictions
│   │   └── semantic_masks_overlay/
│   │       └── {stem}.tif
│   │
│   └── combined/
│       └── sam_dino/
│           ├── combined/
│           │   ├── masks/
│           │   │   └── {stem}_combined_mask.tif
│           │   └── overlays/
│           │       └── {stem}_overlay.tif
│           │
│           ├── Xylem_vessels/           # (with --save-extracted)
│           │   ├── images/
│           │   └── masks/
│           │
│           ├── Hydrated_Xylem_vessels/
│           │   ├── images/
│           │   └── masks/
│           │
│           └── Dehydrated_Xylem_vessels/
│               ├── images/
│               └── masks/
```

## Class Definitions

| Class ID | Class Name | Color (Overlay) |
|----------|------------|-----------------|
| 0 | Background | Gray |
| 1 | Cortex | Blue |
| 2 | Phloem Fibers | Green |
| 3 | Phloem | Purple |
| 4 | Hydrated Xylem vessels | Red |
| 5 | Air-based Pith cells | Yellow |
| 6 | Water-based Pith cells | Orange |
| 7 | Dehydrated Xylem vessels | Teal |

> **Note**: SAM3 prompts Hydrated and Dehydrated Xylem vessels as separate classes (combined IDs 4 and 7). DINO maps both to a single semantic class (ID 4). The `combine_sam_dino_v3` step separates them by subtracting the dilated Dehydrated mask from the Hydrated prediction before applying the DINO spatial filter.

## Key Features

### SAM3 Inference (`src/inference_v6.py`)
- Multi-GPU distributed inference with image-level sharding
- Patch-based processing with configurable overlap (default 50% in pipeline script)
- Max confidence stitching for overlap regions
- Text-prompted segmentation for multiple classes
- Batch processing with native SAM3 API
- Configurable confidence threshold per prompt

### DINO Inference (`src/inference_dino_v1.py`)
- Distributed semantic segmentation across 4 GPUs
- Fine-tuned model support via `lightly_train`
- Percentile-based image normalization (2nd–98th percentile)
- 16-bit / float TIFF input support
- Optional color overlay output

### SAM3 + DINO Combination (`src/combine_sam_dino_v3.py`)
- **`--dino-trust`**: Specified classes are taken directly from DINO semantic mask (no SAM3 mask required)
- **Non-trusted classes**: SAM3 binary masks are filtered spatially using DINO semantic regions
- **Xylem handling**: Dehydrated mask is dilated by `--dilate-px` pixels before subtraction to isolate Hydrated vessels; both are filtered against DINO's xylem region (class 4)
- **`--save-extracted`**: Saves per-class masked images and binary masks under `Xylem_vessels/`, `Hydrated_Xylem_vessels/`, and `Dehydrated_Xylem_vessels/`
- Multi-core CPU processing via Python `multiprocessing.Pool`
- Combined mask and color overlay saved per image
