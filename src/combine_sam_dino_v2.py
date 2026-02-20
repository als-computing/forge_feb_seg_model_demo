"""
Semantic-Guided Mask Filtering and Combination

Uses whole-image semantic masks to filter patch-based predictions.
Only keeps predictions where ≥threshold% of pixels fall within
the corresponding semantic class region.

Author: ALS Photon Science Computing
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed  # ADDED
from multiprocessing import cpu_count                              # ADDED
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Class Definitions
# =============================================================================
CLASS_COLORS_RGB = {
    "Cortex": (0.0, 0.0, 1.0),                        # Blue
    "Phloem_Fibers": (0.0, 1.0, 0.0),                 # Green
    "Phloem": (128/255, 0.0, 128/255),                 # Purple
    "Hydrated_Xylem_vessels": (1.0, 0.0, 0.0),         # Red
    "Air-based_Pith_cells": (1.0, 1.0, 0.0),           # Yellow
    "Water-based_Pith_cells": (1.0, 165/255, 0.0),     # Orange
    "Dehydrated_Xylem_vessels": (1.0, 102/255, 178/255), # Pink
}

# Map prompt names to semantic mask class IDs
# Semantic mask has 6 classes:
#   0: background, 1: Cortex, 2: Phloem Fibers, 3: Phloem,
#   4: Xylem vessels, 5: Air-based Pith cells, 6: Water-based Pith cells
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
def load_mask(mask_path: Path) -> np.ndarray:
    """Load a mask from TIFF file."""
    img = Image.open(mask_path)
    mask = np.array(img)
    
    # Normalize to [0, 1] if needed
    if mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0
    
    return mask


def load_semantic_mask(semantic_path: Path) -> np.ndarray:
    """Load semantic mask (class IDs: 0-6)."""
    img = Image.open(semantic_path)
    return np.array(img).astype(np.uint8)


def filter_mask_by_semantic(
    prediction_mask: np.ndarray,
    semantic_mask: np.ndarray,
    semantic_class_id: int,
) -> np.ndarray:
    """
    Filter prediction mask to ONLY keep pixels within the semantic class region.
    """
    semantic_region = semantic_mask == semantic_class_id
    filtered = prediction_mask.copy()
    filtered[~semantic_region] = 0
    return filtered


def create_combined_mask_from_filtered(
    filtered_masks: Dict[str, np.ndarray],
    prompt_to_class_id: Dict[str, int],
) -> np.ndarray:
    """
    Create combined mask from filtered individual masks.
    Uses max confidence strategy for overlapping regions.
    """
    h, w = next(iter(filtered_masks.values())).shape
    combined = np.zeros((h, w), dtype=np.uint8)
    max_values = np.zeros((h, w), dtype=np.float32)
    
    for prompt, mask in filtered_masks.items():
        class_id = prompt_to_class_id[prompt]
        update_mask = mask > max_values
        combined[update_mask] = class_id
        max_values[update_mask] = mask[update_mask]
    
    return combined


def create_overlay(
    combined_mask: np.ndarray,
    original_image_path: Path,
    class_colors: Dict[str, Tuple[float, float, float]],
    prompt_to_class_id: Dict[str, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Create overlay visualization on original image.
    """
    img = Image.open(original_image_path)
    
    if img.mode not in ["RGB", "L", "RGBA"]:
        if img.mode in ["I", "I;16", "F"]:
            img_array = np.array(img).astype(np.float32)
            p_low = np.percentile(img_array, 0.5)
            p_high = np.percentile(img_array, 99.5)
            img_array = np.clip(img_array, p_low, p_high)
            denom = (p_high - p_low) if (p_high - p_low) > 1e-6 else 1.0
            img_array = ((img_array - p_low) / denom * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        else:
            img = img.convert("L")
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img_array = np.array(img)
    h, w = combined_mask.shape
    
    if img_array.shape[:2] != (h, w):
        img = img.resize((w, h), Image.Resampling.BILINEAR)
        img_array = np.array(img)
    
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    class_id_to_prompt = {v: k for k, v in prompt_to_class_id.items()}
    
    for class_id in np.unique(combined_mask):
        if class_id == 0:
            continue
        
        prompt = class_id_to_prompt.get(class_id)
        if prompt is None or prompt not in class_colors:
            continue
        
        color = class_colors[prompt]
        mask_region = combined_mask == class_id
        
        overlay[mask_region, 0] = color[0]
        overlay[mask_region, 1] = color[1]
        overlay[mask_region, 2] = color[2]
        overlay[mask_region, 3] = alpha
    
    alpha_channel = overlay[:, :, 3:4]
    result = img_array * (1 - alpha_channel) + overlay[:, :, :3] * 255 * alpha_channel
    return result.astype(np.uint8)


def process_image(
    image_stem: str,
    mask_dir: Path,
    semantic_dir: Path,
    input_dir: Path,
    output_dir: Path,
    prompts: List[str],
    prompt_to_semantic: Dict[str, int],
    class_colors: Dict[str, Tuple[float, float, float]],
    alpha: float,
) -> None:
    """
    Process a single image: filter masks, combine, and create overlay.
    """
    semantic_path = semantic_dir / f"{image_stem}.tif"
    if not semantic_path.exists():
        semantic_path = semantic_dir / f"{image_stem}.tiff"
    
    if not semantic_path.exists():
        logger.warning(f"Semantic mask not found for {image_stem}, skipping")
        return
    
    semantic_mask = load_semantic_mask(semantic_path)
    h, w = semantic_mask.shape
    
    filtered_masks = {}
    
    for prompt in prompts:
        prompt_dir = mask_dir / prompt
        mask_path = prompt_dir / f"{image_stem}_mask.tif"
        
        if not mask_path.exists():
            mask_path = prompt_dir / f"{image_stem}_mask.tiff"
        
        if not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}")
            filtered_masks[prompt] = np.zeros((h, w), dtype=np.float32)
            continue
        
        pred_mask = load_mask(mask_path)
        
        if pred_mask.shape != (h, w):
            pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            pred_mask_img = pred_mask_img.resize((w, h), Image.Resampling.BILINEAR)
            pred_mask = np.array(pred_mask_img).astype(np.float32) / 255.0
        
        semantic_class = prompt_to_semantic.get(prompt)
        if semantic_class is None:
            logger.warning(f"No semantic class mapping for {prompt}")
            filtered_masks[prompt] = pred_mask
            continue
        
        filtered = filter_mask_by_semantic(pred_mask, semantic_mask, semantic_class)
        filtered_masks[prompt] = filtered
    
    prompt_to_class_id = {p: i + 1 for i, p in enumerate(prompts)}
    combined = create_combined_mask_from_filtered(filtered_masks, prompt_to_class_id)
    
    orig_path = input_dir / f"{image_stem}.tif"
    if not orig_path.exists():
        orig_path = input_dir / f"{image_stem}.tiff"
    
    if orig_path.exists():
        overlay = create_overlay(combined, orig_path, class_colors, prompt_to_class_id, alpha)
        overlay_path = output_dir / f"{image_stem}_overlay.tif"
        Image.fromarray(overlay).save(overlay_path, compression="tiff_deflate")
    else:
        logger.warning(f"Original image not found: {orig_path}")


# ADDED: wrapper function for multiprocessing
def _process_image_wrapper(args):
    """Wrapper to unpack args for ProcessPoolExecutor."""
    image_stem, mask_dir, semantic_dir, input_dir, output_dir, prompts, prompt_to_semantic, class_colors, alpha = args
    try:
        process_image(
            image_stem=image_stem,
            mask_dir=mask_dir,
            semantic_dir=semantic_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            prompts=prompts,
            prompt_to_semantic=prompt_to_semantic,
            class_colors=class_colors,
            alpha=alpha,
        )
        return (True, image_stem)
    except Exception as e:
        logger.error(f"Error processing {image_stem}: {e}")
        return (False, image_stem)


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Filter and combine masks using semantic segmentation guidance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing original TIFF images",
    )
    parser.add_argument(
        "--instance-masks-dir",
        type=str,
        required=True,
        help="Directory containing prediction masks (with subdirs per prompt)",
    )
    parser.add_argument(
        "--semantic-masks-dir",
        type=str,
        required=True,
        help="Directory containing semantic masks (class IDs 0-6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./filtered_output",
        help="Output directory",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay transparency (0=invisible, 1=opaque)",
    )
    parser.add_argument(                                          # ADDED
        "--num-workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Number of parallel worker processes",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    mask_dir = Path(args.instance_masks_dir)
    semantic_dir = Path(args.semantic_masks_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exclude_dirs = {'combined', 'combined_overlay', 'filtered_combined', 'filtered_overlay'}
    prompts = []
    if mask_dir.exists():
        prompts = [
            d.name for d in mask_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.') and d.name not in exclude_dirs
        ]
        prompts = sorted(prompts)
    
    if not prompts:
        logger.error(f"No prompt subdirectories found in {mask_dir}")
        return
    
    logger.info(f"{'='*70}")
    logger.info("Semantic-Guided Mask Filtering")
    logger.info(f"{'='*70}")
    logger.info(f"  input dir: {input_dir}")
    logger.info(f"  instance masks dir: {mask_dir}")
    logger.info(f"  semantic masks dir: {semantic_dir}")
    logger.info(f"  output dir: {output_dir}")
    logger.info(f"  discovered prompts: {prompts}")
    logger.info(f"  filtering mode: SPATIAL (keep only pixels in semantic region)")
    logger.info(f"  num workers: {args.num_workers}")           # ADDED
    logger.info(f"{'='*70}")
    
    semantic_files = list(semantic_dir.glob("*.tif")) + list(semantic_dir.glob("*.tiff"))
    
    if not semantic_files:
        logger.error(f"No semantic masks found in {semantic_dir}")
        return
    
    logger.info(f"Found {len(semantic_files)} semantic masks to process")
    
    # ADDED: build task list and run in parallel
    tasks = [
        (
            semantic_path.stem,
            mask_dir, semantic_dir, input_dir, output_dir,
            prompts, PROMPT_TO_SEMANTIC_CLASS, CLASS_COLORS_RGB, args.alpha,
        )
        for semantic_path in semantic_files
    ]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_process_image_wrapper, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            future.result()
    
    logger.info(f"{'='*70}")
    logger.info("Processing complete!")
    logger.info(f"  overlays saved to: {output_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()