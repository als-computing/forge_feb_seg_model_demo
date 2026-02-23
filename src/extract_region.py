"""
Extract Region from Original Images and Filter Instance Masks

For a given semantic class, this script:
  1. Masks the original image to keep only pixels in the semantic region (rest = black)
  2. Filters the instance/prediction mask to keep only pixels in the semantic region

Results are saved under output_dir/<class_name>/.

Author: ALS Photon Science Computing
"""

import argparse
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROMPT_TO_SEMANTIC_CLASS = {
    "Cortex": 1,
    "Phloem_Fibers": 2,
    "Phloem": 3,
    "Hydrated_Xylem_vessels": 4,
    "Dehydrated_Xylem_vessels": 4,
    "Air-based_Pith_cells": 5,
    "Water-based_Pith_cells": 6,
}

XYLEM_MODES = ["dehydrated", "hydrated", "both", "all"]


# =============================================================================
# Core Functions
# =============================================================================
def load_semantic_mask(semantic_path: Path) -> np.ndarray:
    return np.array(Image.open(semantic_path)).astype(np.uint8)


def load_instance_mask(path: Path) -> np.ndarray:
    """Load instance mask as float32 [0, 1]."""
    arr = np.array(Image.open(path))
    if arr.max() > 1:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=bool)
    return binary_dilation(mask > 0, structure=struct)


def apply_semantic_filter(mask: np.ndarray, semantic_mask: np.ndarray, class_id: int) -> np.ndarray:
    filtered = mask.copy()
    filtered[semantic_mask != class_id] = 0
    return filtered


def mask_image(image_path: Path, region: np.ndarray) -> np.ndarray:
    img = np.array(Image.open(image_path))
    result = img.copy()
    result[~(region > 0)] = 0
    return result


def save_tif(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path, compression="tiff_deflate")


def find_mask(d: Optional[Path], stem: str) -> Optional[Path]:
    if d is None:
        return None
    for ext in ("_mask.tif", "_mask.tiff"):
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# =============================================================================
# Per-image processing
# =============================================================================
def process_file_wrapper(args: Tuple) -> None:
    (image_stem, image_path, semantic_path,
     hydrated_mask_path, dehydrated_mask_path,
     class_id, out_image_dir, out_mask_dir,
     mode_label, dilate_px) = args
    try:
        sem = load_semantic_mask(semantic_path)

        hydrated_inst   = load_instance_mask(hydrated_mask_path)  if hydrated_mask_path  else None
        dehydrated_inst = load_instance_mask(dehydrated_mask_path) if dehydrated_mask_path else None

        if mode_label == "Xylem_vessels":
            # OR of both binary instance masks, filtered by class 4
            if hydrated_inst is None and dehydrated_inst is None:
                logger.warning(f"No instance masks found for {image_stem}, skipping")
                return
            parts = [m for m in [hydrated_inst, dehydrated_inst] if m is not None]
            union = parts[0] if len(parts) == 1 else ((parts[0] > 0) | (parts[1] > 0)).astype(np.float32)
            mask = apply_semantic_filter(union, sem, class_id)

        elif mode_label == "Dehydrated_Xylem_vessels":
            if dehydrated_inst is None:
                logger.warning(f"Dehydrated instance mask not found for {image_stem}, skipping")
                return
            mask = apply_semantic_filter(dehydrated_inst, sem, class_id)

        elif mode_label == "Hydrated_Xylem_vessels":
            # Hydrated instance mask (contains both) minus dilated dehydrated, filtered by class 4
            if hydrated_inst is None:
                logger.warning(f"Hydrated instance mask not found for {image_stem}, skipping")
                return
            mask = hydrated_inst.copy()
            if dehydrated_inst is not None:
                exclude = dilate_mask(dehydrated_inst, dilate_px)
                mask[exclude] = 0
            else:
                logger.warning(f"Dehydrated instance mask not found for {image_stem}, skipping subtraction")
            mask = apply_semantic_filter(mask, sem, class_id)

        else:
            # Generic single-class (non-xylem): hydrated_inst slot holds the instance mask
            if hydrated_inst is None:
                logger.warning(f"Instance mask not found for {image_stem}, skipping mask filtering")
                mask = None
            else:
                mask = apply_semantic_filter(hydrated_inst, sem, class_id)

        # Save mask
        if mask is not None:
            save_tif((mask * 255).astype(np.uint8), out_mask_dir / f"{image_stem}_mask.tif")

        # Save masked image
        region = mask if mask is not None else (sem == class_id).astype(np.float32)
        save_tif(mask_image(image_path, region), out_image_dir / f"{image_stem}.tif")

    except Exception as e:
        logger.error(f"Error processing {image_stem}: {e}")


# =============================================================================
# Task building & execution
# =============================================================================
def build_tasks(image_files, semantic_dir, instance_root, class_id,
                out_image_dir, out_mask_dir, mode_label, dilate_px):
    hydrated_dir   = instance_root / "Hydrated_Xylem_vessels"   if instance_root else None
    dehydrated_dir = instance_root / "Dehydrated_Xylem_vessels" if instance_root else None

    tasks = []
    for image_path in image_files:
        stem = image_path.stem

        semantic_path = semantic_dir / f"{stem}.tif"
        if not semantic_path.exists():
            semantic_path = semantic_dir / f"{stem}.tiff"
        if not semantic_path.exists():
            logger.warning(f"Semantic mask not found for {stem}, skipping")
            continue

        tasks.append((
            stem, image_path, semantic_path,
            find_mask(hydrated_dir, stem),
            find_mask(dehydrated_dir, stem),
            class_id, out_image_dir, out_mask_dir,
            mode_label, dilate_px,
        ))
    return tasks


def run_extraction(label, class_id, instance_root, image_files,
                   semantic_dir, output_dir, num_workers, dilate_px=0):
    out_image_dir = output_dir / label / "images"
    out_mask_dir  = output_dir / label / "masks"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*70}")
    logger.info(f"  label: {label} (class_id={class_id}, dilate_px={dilate_px})")
    logger.info(f"{'='*70}")

    tasks = build_tasks(image_files, semantic_dir, instance_root,
                        class_id, out_image_dir, out_mask_dir, label, dilate_px)

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_wrapper, tasks), total=len(tasks), desc=f"Processing [{label}]"))

    logger.info(f"Done. Results saved to: {output_dir / label}")


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract semantic region from images and filter instance masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--semantic-masks-dir", type=str, required=True)
    parser.add_argument("--instance-masks-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./extracted_regions")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", type=str, default=None,
                        choices=list(PROMPT_TO_SEMANTIC_CLASS.keys()))
    parser.add_argument("--xylem-mode", type=str, default=None, choices=XYLEM_MODES,
                        help="'dehydrated', 'hydrated', 'both', or 'all'")
    parser.add_argument("--dilate-px", type=int, default=8,
                        help="Pixels to dilate dehydrated mask before subtracting (removes boundary ring)")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    input_dir     = Path(args.input_dir)
    semantic_dir  = Path(args.semantic_masks_dir)
    output_dir    = Path(args.output_dir)
    instance_root = Path(args.instance_masks_dir) if args.instance_masks_dir else None
    num_workers   = args.num_workers or max(1, cpu_count() - 1)

    image_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    if not image_files:
        logger.error(f"No TIFF images found in {input_dir}")
        return
    logger.info(f"Found {len(image_files)} images | workers={num_workers}")

    # -------------------------------------------------------------------------
    # Xylem mode
    # -------------------------------------------------------------------------
    if args.xylem_mode is not None:
        if instance_root is None:
            parser.error("--xylem-mode requires --instance-masks-dir")

        class_id = PROMPT_TO_SEMANTIC_CLASS["Dehydrated_Xylem_vessels"]  # == 4
        mode = args.xylem_mode

        if mode == "all":
            run_extraction("Xylem_vessels", class_id, instance_root,
                           image_files, semantic_dir, output_dir, num_workers, dilate_px=0)

        if mode in ("dehydrated", "both", "all"):
            run_extraction("Dehydrated_Xylem_vessels", class_id, instance_root,
                           image_files, semantic_dir, output_dir, num_workers, dilate_px=0)

        if mode in ("hydrated", "both", "all"):
            run_extraction("Hydrated_Xylem_vessels", class_id, instance_root,
                           image_files, semantic_dir, output_dir, num_workers,
                           dilate_px=args.dilate_px)
        return

    # -------------------------------------------------------------------------
    # Original single-class behaviour
    # -------------------------------------------------------------------------
    if args.class_name is not None:
        class_id    = PROMPT_TO_SEMANTIC_CLASS[args.class_name]
        class_label = args.class_name
    elif args.class_id is not None:
        class_id    = args.class_id
        class_label = next((k for k, v in PROMPT_TO_SEMANTIC_CLASS.items() if v == class_id), str(class_id))
    else:
        parser.error("Provide --class-id, --class-name, or --xylem-mode")

    instance_dir = None
    if instance_root:
        candidate = instance_root / class_label
        if not candidate.exists():
            candidate = instance_root / class_label.replace(" ", "_")
        instance_dir = candidate

    out_image_dir = output_dir / class_label / "images"
    out_mask_dir  = output_dir / class_label / "masks"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*70}")
    logger.info(f"  class: {class_label} (ID={class_id})")
    logger.info(f"  input dir: {input_dir}")
    logger.info(f"  semantic masks dir: {semantic_dir}")
    logger.info(f"  instance masks dir: {instance_dir}")
    logger.info(f"  output dir: {output_dir}")
    logger.info(f"{'='*70}")

    tasks = []
    for image_path in image_files:
        stem = image_path.stem
        semantic_path = semantic_dir / f"{stem}.tif"
        if not semantic_path.exists():
            semantic_path = semantic_dir / f"{stem}.tiff"
        if not semantic_path.exists():
            logger.warning(f"Semantic mask not found for {stem}, skipping")
            continue
        tasks.append((
            stem, image_path, semantic_path,
            find_mask(instance_dir, stem),  # hydrated_mask_path slot (generic)
            None,                           # dehydrated_mask_path
            class_id, out_image_dir, out_mask_dir,
            class_label, 0,
        ))

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_wrapper, tasks), total=len(tasks), desc="Processing"))

    logger.info(f"Done. Results saved to: {output_dir / class_label}")


if __name__ == "__main__":
    main()