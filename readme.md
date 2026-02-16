# Multi-Model Segmentation Pipeline

A comprehensive segmentation pipeline combining SAM3, Cellpose, and DINO models for biological image analysis.

## Overview

This pipeline performs semantic and instance segmentation on microscopy images using three state-of-the-art models:
- **SAM3**: Segment Anything Model 3 for text-prompted segmentation
- **Cellpose**: Deep learning model for cell instance segmentation
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
./run_single_node_all.sh
```

This script will:
1. Run SAM3 inference (uses `sam3_demo` environment)
2. Run Cellpose inference (uses `dino_demo` environment)
3. Run DINO inference (uses `dino_demo` environment)
4. Combine Cellpose + DINO results
5. Combine SAM3 + DINO results

**Note**: The three model inferences can be parallelized in production deployment. The combination steps are CPU-based and do not require GPU.

### Run Individual Tasks

#### SAM3 Inference Only

**Single Node:**
```bash
./run_single_node_v5_ft.sh
```

**Multiple Nodes (Slurm):**
```bash
chmod +x submit_nersc_v4.sh
sbatch submit_nersc_v4.sh
squeue -u $USER
```

## Pipeline Architecture

### Stage 1: Model Inference (Parallel)
- **SAM3**: Text-prompted segmentation for multiple anatomical classes
- **Cellpose**: Instance segmentation for cell detection
- **DINO**: Semantic segmentation for tissue classification

### Stage 2: Result Combination (CPU-based)
- **Cellpose + DINO**: Assigns semantic class labels to detected cell instances
- **SAM3 + DINO**: Filters SAM3 predictions using semantic mask guidance

## Output Structure

After running the pipeline, results are organized as follows:

```
results/
│
├── data4_envtest_YYYYMMDD_HHMMSS/
│   │
│   ├── sam3/
│   │   ├── Cortex/
│   │   │   ├── image1_mask.tif
│   │   │   ├── image2_mask.tif
│   │   │   └── ...
│   │   │
│   │   ├── Phloem/
│   │   │   ├── image1_mask.tif
│   │   │   └── ...
│   │   │
│   │   ├── Phloem_Fibers/
│   │   │   └── ...
│   │   │
│   │   ├── Air-based_Pith_cells/
│   │   │   └── ...
│   │   │
│   │   ├── Water-based_Pith_cells/
│   │   │   └── ...
│   │   │
│   │   ├── Xylem_vessels/
│   │   │   └── ...
│   │   │
│   │   ├── combined/
│   │   │   ├── image1_combined.tif      # Multi-class combined mask
│   │   │   ├── image2_combined.tif
│   │   │   └── ...
│   │   │
│   │   └── combined_overlay/
│   │       ├── image1_overlay.tif       # Visualization overlay
│   │       ├── image2_overlay.tif
│   │       └── ...
│   │
│   ├── cellpose/
│   │   ├── instance_masks/
│   │   │   ├── image1_mask.tif          # Instance segmentation masks
│   │   │   ├── image2_mask.tif
│   │   │   └── ...
│   │   │
│   │   └── instance_masks_overlay/
│   │       ├── image1_overlay.png       # Colored instance overlay
│   │       ├── image2_overlay.png
│   │       └── ...
│   │
│   ├── dino/
│   │   ├── semantic_masks/
│   │   │   ├── image1.tif               # Semantic class predictions
│   │   │   ├── image2.tif
│   │   │   └── ...
│   │   │
│   │   └── semantic_masks_overlay/
│   │       ├── image1.tif               # Semantic overlay visualization
│   │       ├── image2.tif
│   │       └── ...
│   │
│   └── combined/
│       ├── cellpose_dino/
│       │   ├── image1_overlay.png       # Cellpose instances + DINO semantics
│       │   ├── image2_overlay.png
│       │   └── ...
│       │
│       └── sam_dino/
│           ├── image1_overlay.tif       # SAM3 + DINO filtered results
│           ├── image2_overlay.tif
│           └── ...
```

### Output File Descriptions

#### SAM3 Output
- **Class-specific folders**: Binary masks for each anatomical class
- **combined/**: Multi-class segmentation masks (class IDs: 1-6)
- **combined_overlay/**: Color-coded visualization overlays

#### Cellpose Output
- **instance_masks/**: Instance segmentation with unique IDs per cell
- **instance_masks_overlay/**: Color-coded instance visualization

#### DINO Output
- **semantic_masks/**: Pixel-wise class predictions (grayscale, class IDs)
- **semantic_masks_overlay/**: Color-coded semantic visualization

#### Combined Results
- **cellpose_dino/**: Cell instances labeled with semantic classes
- **sam_dino/**: SAM3 predictions filtered by semantic constraints

## Class Definitions

The pipeline segments the following anatomical classes:

| Class ID | Class Name | Color (Overlay) |
|----------|------------|-----------------|
| 0 | Background | Gray |
| 1 | Cortex | Blue |
| 2 | Phloem Fibers | Green |
| 3 | Phloem | Purple |
| 4 | Xylem vessels | Red |
| 5 | Air-based Pith cells | Yellow |
| 6 | Water-based Pith cells | Orange |

## Key Features

### SAM3 Inference (`src/inference_v5.py`)
- Multi-GPU distributed inference with image-level sharding
- Patch-based processing with 25% overlap for large images
- Max confidence stitching for overlap regions
- Text-prompted segmentation for multiple classes
- Batch processing with native SAM3 API

### Cellpose Inference (`src/inference_cellpose_v1.py`)
- Distributed GPU inference
- Custom trained model support
- Instance mask generation with unique cell IDs
- Optional overlay visualization

### DINO Inference (`src/inference_dino_v1.py`)
- Distributed semantic segmentation
- Fine-tuned model support
- Percentile-based image normalization
- Batch processing with multi-GPU support

### Result Combination

**Cellpose + DINO** (`src/combine_cellpose_dino.py`):
- Assigns semantic class labels to cell instances
- Uses majority voting with 30% overlap threshold
- Multi-core CPU processing for efficiency

**SAM3 + DINO** (`src/combine_sam_dino.py`):
- Filters SAM3 predictions using semantic mask guidance
- Spatial constraint: only keeps predictions within semantic regions
- Max confidence strategy for overlapping predictions
- Reduces false positives from SAM3

