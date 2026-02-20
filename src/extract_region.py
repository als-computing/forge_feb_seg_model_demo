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
from typing import Tuple

import numpy as np
from PIL import Image
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
    "Hydrated_Xylem_vessels": 4,       # both xylem types map to class 4
    "Dehydrated_Xylem_vessels": 4,     # both xylem types map to class 4
    "Air-based_Pith_cells": 5,
    "Water-based_Pith_cells": 6,
}


# =============================================================================
# Core Functions
# =============================================================================
def load_semantic_mask(semantic_path: Path) -> np.ndarray:
    """Load semantic mask (class IDs: 0-6)."""
    return np.array(Image.open(semantic_path)).astype(np.uint8)


def filter_mask_by_semantic(
    prediction_mask: np.ndarray,
    semantic_mask: np.ndarray,
    semantic_class_id: int,
) -> np.ndarray:
    """
    Filter prediction mask to ONLY keep pixels within the semantic class region.

    Args:
        prediction_mask: Prediction mask [0, 1]
        semantic_mask: Semantic segmentation [0-6]
        semantic_class_id: Expected class ID in semantic mask

    Returns:
        Filtered mask with predictions only in semantic region
    """
    semantic_region = semantic_mask == semantic_class_id
    filtered = prediction_mask.copy()
    filtered[~semantic_region] = 0
    return filtered


def extract_image_region(
    image_path: Path,
    semantic_mask: np.ndarray,
    class_id: int,
) -> np.ndarray:
    """Load original image and zero out pixels outside the semantic class region."""
    img_array = np.array(Image.open(image_path))
    h, w = img_array.shape[:2]

    if semantic_mask.shape != (h, w):
        semantic_mask = np.array(
            Image.fromarray(semantic_mask).resize((w, h), Image.Resampling.NEAREST)
        )

    region = semantic_mask == class_id
    result = img_array.copy()
    result[~region] = 0
    return result


def process_image(
    image_stem: str,
    image_path: Path,
    semantic_mask: np.ndarray,
    instance_mask_path: Path,
    class_id: int,
    out_image_dir: Path,
    out_mask_dir: Path,
) -> None:
    """Process a single image: extract region and filter instance mask."""
    # 1. Masked original image
    masked_image = extract_image_region(image_path, semantic_mask, class_id)
    Image.fromarray(masked_image).save(
        out_image_dir / f"{image_stem}.tif", compression="tiff_deflate"
    )

    # 2. Filtered instance mask (if available)
    if instance_mask_path is not None and instance_mask_path.exists():
        pred = np.array(Image.open(instance_mask_path))
        if pred.max() > 1:
            pred = pred.astype(np.float32) / 255.0
        else:
            pred = pred.astype(np.float32)

        filtered = filter_mask_by_semantic(pred, semantic_mask, class_id)
        filtered_uint8 = (filtered * 255).astype(np.uint8)
        Image.fromarray(filtered_uint8).save(
            out_mask_dir / f"{image_stem}_mask.tif", compression="tiff_deflate"
        )
    else:
        logger.warning(f"Instance mask not found for {image_stem}, skipping mask filtering")


def process_file_wrapper(args: Tuple) -> None:
    """Wrapper for multiprocessing — unpacks tuple and calls process_image."""
    image_stem, image_path, semantic_path, instance_mask_path, class_id, out_image_dir, out_mask_dir = args
    try:
        semantic_mask = load_semantic_mask(semantic_path)
        process_image(
            image_stem=image_stem,
            image_path=image_path,
            semantic_mask=semantic_mask,
            instance_mask_path=instance_mask_path,
            class_id=class_id,
            out_image_dir=out_image_dir,
            out_mask_dir=out_mask_dir,
        )
    except Exception as e:
        logger.error(f"Error processing {image_stem}: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract semantic region from images and filter instance masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing original TIFF images")
    parser.add_argument("--semantic-masks-dir", type=str, required=True,
                        help="Directory containing semantic masks (class IDs 0-6)")
    parser.add_argument("--instance-masks-dir", type=str, default=None,
                        help="Root directory containing instance masks (subdir per class name is used)")
    parser.add_argument("--output-dir", type=str, default="./extracted_regions",
                        help="Output directory")
    parser.add_argument("--class-id", type=int, default=None,
                        help="Semantic class ID to extract (e.g. 4 for Xylem vessels)")
    parser.add_argument("--class-name", type=str, default=None,
                        choices=list(PROMPT_TO_SEMANTIC_CLASS.keys()),
                        help="Semantic class name to extract (alternative to --class-id)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of CPU workers (default: cpu_count - 1)")
    args = parser.parse_args()

    # Resolve class ID and label FIRST
    if args.class_name is not None:
        class_id = PROMPT_TO_SEMANTIC_CLASS[args.class_name]
        class_label = args.class_name
    elif args.class_id is not None:
        class_id = args.class_id
        class_label = next((k for k, v in PROMPT_TO_SEMANTIC_CLASS.items() if v == class_id), str(class_id))
    else:
        parser.error("Provide either --class-id or --class-name")

    input_dir = Path(args.input_dir)
    semantic_dir = Path(args.semantic_masks_dir)
    output_dir = Path(args.output_dir)
    if args.instance_masks_dir:
        instance_dir = Path(args.instance_masks_dir) / class_label
        if not instance_dir.exists():
            # Fallback: try with underscores instead of spaces
            instance_dir = Path(args.instance_masks_dir) / class_label.replace(" ", "_")
    else:
        instance_dir = None

    # Output subdirectories named after the class
    out_image_dir = output_dir / class_label / "images"
    out_mask_dir = output_dir / class_label / "masks"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*70}")
    logger.info(f"  class: {class_label} (ID={class_id})")
    logger.info(f"  input dir: {input_dir}")
    logger.info(f"  semantic masks dir: {semantic_dir}")
    logger.info(f"  instance masks dir: {instance_dir}")
    logger.info(f"  output dir: {output_dir}")
    logger.info(f"{'='*70}")

    image_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    if not image_files:
        logger.error(f"No TIFF images found in {input_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process")

    # Build task list
    tasks = []
    for image_path in image_files:
        stem = image_path.stem

        semantic_path = semantic_dir / f"{stem}.tif"
        if not semantic_path.exists():
            semantic_path = semantic_dir / f"{stem}.tiff"
        if not semantic_path.exists():
            logger.warning(f"Semantic mask not found for {stem}, skipping")
            continue

        instance_mask_path = None
        if instance_dir is not None:
            instance_mask_path = instance_dir / f"{stem}_mask.tif"
            if not instance_mask_path.exists():
                instance_mask_path = instance_dir / f"{stem}_mask.tiff"

        tasks.append((stem, image_path, semantic_path, instance_mask_path, class_id, out_image_dir, out_mask_dir))

    # Multi-CPU processing
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    logger.info(f"  num workers: {num_workers}")

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_wrapper, tasks), total=len(tasks), desc="Processing"))

    logger.info(f"Done. Results saved to: {output_dir / class_label}")


if __name__ == "__main__":
    main()