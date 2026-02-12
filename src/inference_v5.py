"""
SAM3 Distributed Segmentation Inference - Native Batching Version (IMAGE-SHARDED)
WITH MAX CONFIDENCE STITCHING

Key changes:
✅ 25% overlap in patching
✅ Max confidence stitching for overlap regions
✅ Max confidence for combined masks (multi-class conflicts)
✅ SHARD BY IMAGE ACROSS RANKS (no patch-level distributed gather)


Author: ALS Photon Science Computing
"""

import os
import sys
from pathlib import Path

# =============================================================================
# Environment Setup (must happen before other imports)
# =============================================================================
from dotenv import load_dotenv
load_dotenv()

if "HF_HUB_CACHE" in os.environ:
    os.environ["HF_HOME"] = os.environ["HF_HUB_CACHE"]

# =============================================================================
# Imports
# =============================================================================
import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

# SAM3 native imports
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global counter for unique query IDs (reset per image)
GLOBAL_COUNTER = 1

# =============================================================================
# Class Color Definitions
# =============================================================================
CLASS_COLORS_RGB = {
    "Cortex": (0.0, 0.0, 1.0),                    # Blue
    "Phloem Fibers": (0.0, 1.0, 0.0),             # Green
    "Phloem": (128/255, 0.0, 128/255),            # Purple
    "Xylem vessels": (1.0, 0.0, 0.0),             # Red
    "Air-based Pith cells": (1.0, 1.0, 0.0),      # Yellow
    "Water-based Pith cells": (1.0, 165/255, 0.0) # Orange
}


# =============================================================================
# SAM3 Native API Helper Functions
# =============================================================================
def create_empty_datapoint() -> Datapoint:
    """Create an empty datapoint for a single image."""
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint: Datapoint, pil_image: Image.Image) -> None:
    """Attach the image to the datapoint."""
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint: Datapoint, text_query: str) -> int:
    """
    Add a text query to the datapoint.

    Returns:
        Unique query_id for this prompt (used to retrieve results).
    """
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "Please set the image first"

    h, w = datapoint.images[0].size  # [h, w]

    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1


# =============================================================================
# Image Processing Utilities
# =============================================================================
def load_state_dict_flexible(ckpt_path: str) -> dict:
    """Load a checkpoint and return a clean state_dict (handles common nesting)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    return state


def tif_to_rgb_pil(
    image_path: str,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> Image.Image:
    """Convert a TIFF image to RGB PIL Image with percentile normalization for 16-bit types."""
    img = Image.open(image_path)

    if img.mode not in ["RGB", "L", "RGBA"]:
        if img.mode in ["I", "I;16", "F"]:
            img_array = np.array(img).astype(np.float32)
            p_low = np.percentile(img_array, percentile_low)
            p_high = np.percentile(img_array, percentile_high)
            img_array = np.clip(img_array, p_low, p_high)
            denom = (p_high - p_low) if (p_high - p_low) > 1e-6 else 1.0
            img_array = ((img_array - p_low) / denom * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        else:
            img = img.convert("L")

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def crop_image_to_patches(
    image_pil: Image.Image,
    patch_size: int = 512,
    overlap_ratio: float = 0.25,
) -> Tuple[List[Tuple], int, int, int, int]:
    """
    Crop a PIL image into patches with 25% overlap.
    
    Args:
        image_pil: Input PIL image
        patch_size: Size of each patch
        overlap_ratio: Overlap between patches (default 0.25 for 25% overlap)
    
    Returns:
        patches: List of (patch_pil, row, col, y_start, x_start, patch_h, patch_w) tuples
        H, W: Original image dimensions
        n_rows, n_cols: Grid dimensions
    """
    W, H = image_pil.size
    patches: List[Tuple] = []

    # Calculate step size with overlap
    step = int(patch_size * (1 - overlap_ratio))
    
    # Calculate number of patches needed
    n_rows = max(1, (H - patch_size) // step + 1)
    n_cols = max(1, (W - patch_size) // step + 1)
    
    # Ensure we cover the entire image
    if n_rows > 1 and (n_rows - 1) * step + patch_size < H:
        n_rows += 1
    if n_cols > 1 and (n_cols - 1) * step + patch_size < W:
        n_cols += 1

    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate patch position, ensuring we don't go out of bounds
            y = min(row * step, H - patch_size) if H >= patch_size else 0
            x = min(col * step, W - patch_size) if W >= patch_size else 0

            patch_w = min(patch_size, W - x)
            patch_h = min(patch_size, H - y)

            patch = image_pil.crop((x, y, x + patch_w, y + patch_h))

            # Pad undersized patches
            if patch_w < patch_size or patch_h < patch_size:
                padded = Image.new("RGB", (patch_size, patch_size), (0, 0, 0))
                padded.paste(patch, (0, 0))
                patch = padded

            patches.append((patch, row, col, y, x, patch_h, patch_w))

    return patches, H, W, n_rows, n_cols


def list_tiff_files(input_dir: str) -> List[Path]:
    """List TIFF files (paths only)."""
    input_path = Path(input_dir)
    tiff_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
    return tiff_files


def shard_indices(n: int, rank: int, world_size: int) -> range:
    """
    Deterministic contiguous sharding:
    - rank 0 gets [0, ..., k0)
    - rank 1 gets [k0, ..., k1)
    ...
    Balanced within 1.
    """
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return range(start, end)


# =============================================================================
# Distributed Setup (minimal: only for rank/world_size discovery + optional final metrics)
# =============================================================================
def setup_distributed() -> Tuple[int, int, int]:
    """Initialize PyTorch distributed environment (NCCL) if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set device BEFORE init to avoid rank->GPU ambiguity warnings on some systems.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        if rank == 0:
            logger.info(f"Distributed initialized: world_size={world_size} ranks")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        logger.info("Running in single-GPU mode")

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Model Loading
# =============================================================================
def load_model(
    bpe_path: str,
    device: torch.device,
    finetuned_checkpoint: Optional[str] = None,
    original_checkpoint: Optional[str] = None,
    rank: int = 0,
) -> torch.nn.Module:
    """Load SAM3 model with optional finetuned weights."""
    if rank == 0:
        logger.info("Loading SAM3 model...")

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
    )

    if finetuned_checkpoint:
        if not original_checkpoint:
            raise ValueError("original_checkpoint required when using finetuned model")

        if rank == 0:
            logger.info(f"  Base checkpoint: {original_checkpoint}")
            logger.info(f"  Finetuned checkpoint: {finetuned_checkpoint}")

        original_state = load_state_dict_flexible(original_checkpoint)
        model.load_state_dict(original_state, strict=False)

        finetuned_state = load_state_dict_flexible(finetuned_checkpoint)
        missing, unexpected = model.load_state_dict(finetuned_state, strict=False)

        if rank == 0:
            if missing:
                logger.warning(f"  Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"  Unexpected keys: {len(unexpected)}")
            logger.info("✓ Finetuned model loaded")

    elif original_checkpoint:
        if rank == 0:
            logger.info(f"  Loading from checkpoint: {original_checkpoint}")
        original_state = load_state_dict_flexible(original_checkpoint)
        model.load_state_dict(original_state, strict=False)
        if rank == 0:
            logger.info("✓ Model loaded from checkpoint")
    else:
        if rank == 0:
            logger.info("✓ Pretrained model loaded")

    model.eval().to(device)
    return model


def create_transform(image_size: int = 1008) -> ComposeAPI:
    """Create the SAM3 preprocessing transform."""
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=image_size, max_size=image_size, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_postprocessor(confidence_threshold: float = 0.5) -> PostProcessImage:
    """Create the SAM3 postprocessor."""
    return PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=confidence_threshold,
        to_cpu=False,
    )


# =============================================================================
# Core Inference with Native Batching (single-rank per image)
# =============================================================================
def process_patches_batch_native(
    patch_data_list: List[Tuple[int, Tuple]],
    prompts: List[str],
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    device: torch.device,
    patch_size: int,
) -> List[dict]:
    """
    Process a batch of patches using SAM3's native batching API.
    All patches AND all prompts are processed in a single forward pass.
    NOW INCLUDES CONFIDENCE SCORES for max confidence stitching.
    """
    global GLOBAL_COUNTER

    datapoints: List[Datapoint] = []
    query_id_map: Dict[int, Tuple] = {}

    for batch_idx, (global_idx, patch_tuple) in enumerate(patch_data_list):
        patch_pil, row, col, y_start, x_start, patch_h, patch_w = patch_tuple

        dp = create_empty_datapoint()
        set_image(dp, patch_pil)

        for prompt in prompts:
            query_id = add_text_prompt(dp, prompt)
            query_id_map[query_id] = (
                batch_idx,
                prompt,
                global_idx,
                row,
                col,
                y_start,
                x_start,
                patch_h,
                patch_w,
            )

        dp = transform(dp)
        datapoints.append(dp)

    batch = collate(datapoints, dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    with torch.inference_mode():
        output = model(batch)

    processed_results = postprocessor.process_results(output, batch.find_metadatas)

    batch_results: Dict[int, dict] = {}
    for batch_idx, (global_idx, patch_tuple) in enumerate(patch_data_list):
        patch_pil, row, col, y_start, x_start, patch_h, patch_w = patch_tuple
        batch_results[batch_idx] = {
            "idx": global_idx,
            "row": row,
            "col": col,
            "y_start": y_start,
            "x_start": x_start,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "prompt_masks": {},
            "prompt_scores": {},
        }

    for query_id, (batch_idx, prompt, *_rest) in query_id_map.items():
        if query_id in processed_results:
            result = processed_results[query_id]
            if "masks" in result and len(result["masks"]) > 0:
                masks = result["masks"]
                scores = result.get("scores", None)

                if torch.is_tensor(masks):
                    masks = masks.cpu().numpy()
                if scores is not None and torch.is_tensor(scores):
                    scores = scores.cpu().numpy()

                batch_results[batch_idx]["prompt_masks"][prompt] = masks
                batch_results[batch_idx]["prompt_scores"][prompt] = scores
            else:
                batch_results[batch_idx]["prompt_masks"][prompt] = None
                batch_results[batch_idx]["prompt_scores"][prompt] = None
        else:
            batch_results[batch_idx]["prompt_masks"][prompt] = None
            batch_results[batch_idx]["prompt_scores"][prompt] = None

    return list(batch_results.values())


def process_patches_native_single_rank(
    patches: List[Tuple],
    prompts: List[str],
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    batch_size: int,
    device: torch.device,
    patch_size: int,
) -> List[dict]:
    """
    Process all patches for ONE IMAGE on ONE RANK.
    No distributed sampler, no collectives.
    """
    results: List[dict] = []
    # Deterministic, stable ordering
    n = len(patches)
    for start in range(0, n, batch_size):
        chunk = patches[start : start + batch_size]
        patch_data_list = [(start + i, chunk[i]) for i in range(len(chunk))]
        batch_results = process_patches_batch_native(
            patch_data_list=patch_data_list,
            prompts=prompts,
            model=model,
            transform=transform,
            postprocessor=postprocessor,
            device=device,
            patch_size=patch_size,
        )
        results.extend(batch_results)
    return results


# =============================================================================
# Post-Processing: Max Confidence Stitching (NEW)
# =============================================================================
def stitch_masks_for_prompt_max_confidence(
    results: List[dict],
    prompt: str,
    orig_h: int,
    orig_w: int,
    patch_size: int,
) -> np.ndarray:
    """
    Stitch patch masks using MAX CONFIDENCE strategy.
    Pixels with highest confidence scores win in overlap regions.
    """
    stitched = np.zeros((orig_h, orig_w), dtype=np.float32)
    max_scores = np.full((orig_h, orig_w), -float('inf'), dtype=np.float32)

    for result in results:
        masks = result["prompt_masks"].get(prompt)
        scores = result["prompt_scores"].get(prompt)
        y_start = result["y_start"]
        x_start = result["x_start"]
        patch_h = result["patch_h"]
        patch_w = result["patch_w"]

        if masks is None or (isinstance(masks, np.ndarray) and masks.size == 0):
            continue

        # Create score map for this patch
        score_map = np.zeros((patch_size, patch_size), dtype=np.float32)
        combined_mask = np.zeros((patch_size, patch_size), dtype=np.float32)

        masks = np.array(masks)
        if scores is not None:
            scores = np.array(scores)
            # Normalize scores if needed
            if scores.size > 0 and (np.nanmax(scores) > 10 or np.nanmin(scores) < -10):
                scores = 1 / (1 + np.exp(-scores))
        else:
            scores = np.ones(len(masks)) * 0.5 if len(masks) > 0 else np.array([])

        # Process each mask with its score
        for i, m in enumerate(masks):
            if m.ndim == 3:
                m = m.squeeze(0)
            
            # Resize if needed
            if m.shape != (patch_size, patch_size):
                m_tensor = torch.from_numpy(m).float()[None, None, ...]
                m_tensor = torch.nn.functional.interpolate(
                    m_tensor,
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                m = m_tensor.squeeze().numpy()
            
            # Update score map and combined mask where this mask is
            mask_binary = m > 0.5
            score = scores[i] if i < len(scores) else 0.5
            
            # Where this mask is better, update both score and mask
            update_mask = mask_binary & (score > score_map)
            score_map[update_mask] = score
            combined_mask[update_mask] = m[update_mask]

        # Crop to actual patch size
        score_map_crop = score_map[:patch_h, :patch_w]
        combined_crop = combined_mask[:patch_h, :patch_w]

        # Update global stitched image where scores are higher
        y_end = y_start + patch_h
        x_end = x_start + patch_w
        
        current_scores = max_scores[y_start:y_end, x_start:x_end]
        update_mask = score_map_crop > current_scores
        
        stitched[y_start:y_end, x_start:x_end][update_mask] = combined_crop[update_mask]
        max_scores[y_start:y_end, x_start:x_end] = np.maximum(current_scores, score_map_crop)

    return stitched


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save a mask as a TIFF file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mask.max() <= 1.0:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)

    Image.fromarray(mask_uint8).save(output_path, compression="tiff_deflate")


def save_combined_mask_max_confidence(
    results: List[dict],
    prompts: List[str],
    class_colors: Dict[str, int],
    orig_h: int,
    orig_w: int,
    patch_size: int,
    output_path: Path,
) -> None:
    """
    Create and save a combined segmentation mask using MAX CONFIDENCE strategy.
    When multiple classes detect the same pixel, the class with highest confidence wins.
    
    Args:
        results: List of patch results with masks and scores for all prompts
        prompts: List of class names
        class_colors: Dict mapping prompt name to class ID
        orig_h, orig_w: Original image dimensions
        patch_size: Size of patches used for inference
        output_path: Where to save the combined mask
    """
    combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    max_scores = np.full((orig_h, orig_w), -float('inf'), dtype=np.float32)
    
    # Process each class
    for prompt in prompts:
        class_id = class_colors[prompt]
        
        # Process each patch
        for result in results:
            masks = result["prompt_masks"].get(prompt)
            scores = result["prompt_scores"].get(prompt)
            y_start = result["y_start"]
            x_start = result["x_start"]
            patch_h = result["patch_h"]
            patch_w = result["patch_w"]
            
            if masks is None or (isinstance(masks, np.ndarray) and masks.size == 0):
                continue
            
            # Create score map for this patch
            score_map = np.zeros((patch_size, patch_size), dtype=np.float32)
            class_mask = np.zeros((patch_size, patch_size), dtype=bool)
            
            masks = np.array(masks)
            if scores is not None:
                scores = np.array(scores)
                # Normalize scores if needed
                if scores.size > 0 and (np.nanmax(scores) > 10 or np.nanmin(scores) < -10):
                    scores = 1 / (1 + np.exp(-scores))
            else:
                scores = np.ones(len(masks)) * 0.5 if len(masks) > 0 else np.array([])
            
            # Process each mask with its score
            for i, m in enumerate(masks):
                if m.ndim == 3:
                    m = m.squeeze(0)
                
                # Resize if needed
                if m.shape != (patch_size, patch_size):
                    m_tensor = torch.from_numpy(m).float()[None, None, ...]
                    m_tensor = torch.nn.functional.interpolate(
                        m_tensor,
                        size=(patch_size, patch_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    m = m_tensor.squeeze().numpy()
                
                # Update score map where this mask is better
                mask_binary = m > 0.5
                score = scores[i] if i < len(scores) else 0.5
                
                update_mask = mask_binary & (score > score_map)
                score_map[update_mask] = score
                class_mask[update_mask] = True
            
            # Crop to actual patch size
            score_map_crop = score_map[:patch_h, :patch_w]
            class_mask_crop = class_mask[:patch_h, :patch_w]
            
            # Update global combined mask where this class has higher confidence
            y_end = y_start + patch_h
            x_end = x_start + patch_w
            
            current_scores = max_scores[y_start:y_end, x_start:x_end]
            update_mask = class_mask_crop & (score_map_crop > current_scores)
            
            combined[y_start:y_end, x_start:x_end][update_mask] = class_id
            max_scores[y_start:y_end, x_start:x_end][update_mask] = score_map_crop[update_mask]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(output_path, compression="tiff_deflate")


def save_overlay(
    results: List[dict],
    prompts: List[str],
    class_colors: Dict[str, int],
    orig_h: int,
    orig_w: int,
    patch_size: int,
    image_path: Path,
    output_path: Path,
    alpha: float = 0.45,
) -> None:
    """
    Create and save an overlay using MAX CONFIDENCE strategy.
    EXACTLY MATCHES NOTEBOOK: Uses RGBA overlay with proper blending.
    
    Args:
        results: List of patch results with masks and scores
        prompts: List of class names
        class_colors: Dict mapping prompt name to class ID
        orig_h, orig_w: Original image dimensions
        patch_size: Size of patches
        image_path: Path to the original image
        output_path: Where to save the overlay
        alpha: Transparency of the overlay (0=invisible, 1=opaque)
    """
    # Load and normalize the original image EXACTLY like notebook
    img = Image.open(image_path)
    
    # Handle different image modes EXACTLY like notebook
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
    
    # Create 4-channel overlay (RGBA) EXACTLY like notebook
    overlay = np.zeros((orig_h, orig_w, 4), dtype=np.float32)
    max_scores = np.full((orig_h, orig_w), -float('inf'), dtype=np.float32)
    
    # Process each class with max confidence strategy EXACTLY like notebook
    for prompt in prompts:
        if prompt not in CLASS_COLORS_RGB:
            continue
        
        class_id = class_colors[prompt]
        color_rgb_normalized = CLASS_COLORS_RGB[prompt]
        
        # Process each patch for this class
        for result in results:
            masks = result["prompt_masks"].get(prompt)
            scores = result["prompt_scores"].get(prompt)
            y_start = result["y_start"]
            x_start = result["x_start"]
            patch_h = result["patch_h"]
            patch_w = result["patch_w"]
            
            if masks is None or (isinstance(masks, np.ndarray) and masks.size == 0):
                continue
            
            masks = np.array(masks)
            if scores is not None:
                scores = np.array(scores)
                # Normalize scores EXACTLY like notebook
                if scores.size > 0 and (np.nanmax(scores) > 10 or np.nanmin(scores) < -10):
                    scores = 1 / (1 + np.exp(-scores))
            else:
                scores = np.ones(len(masks)) * 0.5 if len(masks) > 0 else np.array([])
            
            # Process each mask EXACTLY like notebook
            for i, m in enumerate(masks):
                if m.ndim == 3:
                    m = m.squeeze(0)
                
                score = scores[i] if i < len(scores) else 0.5
                
                # CRITICAL: Resize to patch_size FIRST (like notebook)
                if m.shape != (patch_size, patch_size):
                    m_tensor = torch.from_numpy(m).float()[None, None, ...]
                    m_tensor = torch.nn.functional.interpolate(
                        m_tensor,
                        size=(patch_size, patch_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    m = m_tensor.squeeze().numpy()
                
                # THEN crop to actual patch size (like notebook)
                m_crop = m[:patch_h, :patch_w]
                
                # Update where this class has higher confidence
                mask_binary = m_crop > 0.5
                y_end = y_start + patch_h
                x_end = x_start + patch_w
                
                current_scores = max_scores[y_start:y_end, x_start:x_end]
                update_mask = mask_binary & (score > current_scores)
                
                # Update overlay with RGBA (EXACTLY like notebook)
                overlay[y_start:y_end, x_start:x_end][update_mask, 0] = color_rgb_normalized[0]
                overlay[y_start:y_end, x_start:x_end][update_mask, 1] = color_rgb_normalized[1]
                overlay[y_start:y_end, x_start:x_end][update_mask, 2] = color_rgb_normalized[2]
                overlay[y_start:y_end, x_start:x_end][update_mask, 3] = alpha
                max_scores[y_start:y_end, x_start:x_end][update_mask] = score
    
    # Blend overlay with original image (EXACTLY like notebook)
    alpha_channel = overlay[:, :, 3:4]
    result = img_array * (1 - alpha_channel) + overlay[:, :, :3] * 255 * alpha_channel
    result = result.astype(np.uint8)
    
    # Save the overlay
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_path, compression="tiff_deflate")


def process_and_save_one_image_single_rank(
    image_path: Path,
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    prompts: List[str],
    class_colors: Dict[str, int],
    output_dir: Path,
    patch_size: int,
    overlap_ratio: float,
    batch_size: int,
    device: torch.device,
    rank: int,
) -> float:
    """
    End-to-end processing for ONE IMAGE on ONE RANK:
    load -> patch (with overlap) -> batched inference -> max confidence stitch -> write outputs

    Returns:
        wall time seconds for this image.
    """
    global GLOBAL_COUNTER
    GLOBAL_COUNTER = 1

    t0 = time.time()
    image_pil = tif_to_rgb_pil(str(image_path))
    image_name = image_path.name

    patches, orig_h, orig_w, n_rows, n_cols = crop_image_to_patches(
        image_pil, patch_size, overlap_ratio
    )

    logger.info(
        f"[rank {rank}] Processing {image_name} | size={orig_w}x{orig_h} | "
        f"patches={len(patches)} ({n_rows}x{n_cols}) | overlap={overlap_ratio*100:.0f}%"
    )

    # Inference over all patches (single-rank)
    local_results = process_patches_native_single_rank(
        patches=patches,
        prompts=prompts,
        model=model,
        transform=transform,
        postprocessor=postprocessor,
        batch_size=batch_size,
        device=device,
        patch_size=patch_size,
    )

    # Stitch with MAX CONFIDENCE + save (single-rank)
    image_stem = image_path.stem

    for prompt in prompts:
        stitched = stitch_masks_for_prompt_max_confidence(
            local_results, prompt, orig_h, orig_w, patch_size
        )

        prompt_dir = output_dir / prompt.replace(" ", "_")
        save_mask(stitched, prompt_dir / f"{image_stem}_mask.tif")

    # Create combined mask using MAX CONFIDENCE across all classes
    combined_path = output_dir / "combined" / f"{image_stem}_combined.tif"
    save_combined_mask_max_confidence(
        local_results, prompts, class_colors, orig_h, orig_w, patch_size, combined_path
    )

    # Save overlay
    overlay_path = output_dir / "combined_overlay" / f"{image_stem}_overlay.tif"
    save_overlay(
        results=local_results,
        prompts=prompts,
        class_colors=class_colors,
        orig_h=orig_h,
        orig_w=orig_w,
        patch_size=patch_size,
        image_path=image_path,
        output_path=overlay_path,
        alpha=0.45,
    )

    dt = time.time() - t0
    logger.info(f"[rank {rank}] ✓ Saved masks for {image_name} in {dt:.1f}s")
    return dt


# =============================================================================
# Main Entry Point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM3 Inference with Native Batching + Max Confidence Stitching (IMAGE-SHARDED)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--bpe-path", type=str, required=True)

    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--finetuned-checkpoint", type=str, default=None)
    parser.add_argument("--original-checkpoint", type=str, default=None)

    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap-ratio", type=float, default=0.25, 
                       help="Overlap ratio between patches (0.25 = 25%)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Patches per forward pass (native batching lets you go larger).",
    )
    parser.add_argument(
        "--confidence",
        nargs="+",
        type=float,
        default=[0.5],
        help="Confidence thresholds for each prompt (single value for all or one per prompt)"
    )
    parser.add_argument("--image-size", type=int, default=1008)

    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["cortex", "Phloem Fibers", "Xylem vessels", "Pith cells", "outer cells"],
    )

    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    # Distributed init (for rank/world_size and final metrics only)
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Perf toggles
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle confidence thresholds
    if len(args.confidence) == 1:
        # Use same confidence for all prompts
        confidence_thresholds = args.confidence * len(args.prompts)
    elif len(args.confidence) == len(args.prompts):
        # Use different confidence for each prompt
        confidence_thresholds = args.confidence
    else:
        raise ValueError(
            f"Confidence list length ({len(args.confidence)}) must be either 1 "
            f"or match number of prompts ({len(args.prompts)})"
        )

    # Build the SAME file list on every rank (paths only, cheap)
    all_files = list_tiff_files(args.input_dir)

    # Apply skip-existing deterministically on every rank (based on combined outputs)
    if args.skip_existing:
        combined_dir = output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.stem.replace("_combined", "") for p in combined_dir.glob("*.tif")}
        all_files = [p for p in all_files if p.stem not in existing]

    if rank == 0:
        logger.info(f"{'='*70}")
        logger.info("Starting distributed inference (IMAGE-SHARDED + MAX CONFIDENCE)")
        logger.info(f"{'='*70}")
        logger.info(f"  world_size (ranks): {world_size}")
        logger.info(f"  input dir: {args.input_dir}")
        logger.info(f"  output dir: {output_dir}")
        logger.info(f"  images total (after skip): {len(all_files)}")
        logger.info(f"  prompts: {args.prompts}")
        logger.info(f"  confidence thresholds: {confidence_thresholds}")
        logger.info(f"  patch_size: {args.patch_size} | overlap: {args.overlap_ratio*100:.0f}% | batch_size: {args.batch_size}")
        logger.info(f"  stitching: MAX CONFIDENCE (per-class + combined)")
        logger.info(f"{'='*70}")

    if not all_files:
        if rank == 0:
            logger.info("No images to process. Exiting.")
        cleanup_distributed()
        return

    # Shard by image index (balanced contiguous ranges)
    my_range = shard_indices(len(all_files), rank, world_size)
    my_files = [all_files[i] for i in my_range]

    logger.info(f"[rank {rank}] Assigned {len(my_files)} / {len(all_files)} images")

    # Load model ON EACH RANK (no DDP needed; pure inference)
    model = load_model(
        bpe_path=args.bpe_path,
        device=device,
        finetuned_checkpoint=args.finetuned_checkpoint,
        original_checkpoint=args.original_checkpoint,
        rank=rank,
    )

    transform = create_transform(args.image_size)
    postprocessor = create_postprocessor(args.confidence[0])

    class_colors = {prompt: idx + 1 for idx, prompt in enumerate(args.prompts)}

    # Process our shard
    t_total0 = time.time()
    per_image_times: List[float] = []
    processed_count = 0

    for img_path in my_files:
        dt = process_and_save_one_image_single_rank(
            image_path=img_path,
            model=model,
            transform=transform,
            postprocessor=postprocessor,
            prompts=args.prompts,
            class_colors=class_colors,
            output_dir=output_dir,
            patch_size=args.patch_size,
            overlap_ratio=args.overlap_ratio,
            batch_size=args.batch_size,
            device=device,
            rank=rank,
        )
        per_image_times.append(dt)
        processed_count += 1

    my_total = time.time() - t_total0

    # Optional global summary
    if world_size > 1 and dist.is_initialized():
        # Total images
        t_count = torch.tensor([processed_count], device=device, dtype=torch.long)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)

        # Total time across ranks (sum) and max time (tail)
        t_sum = torch.tensor([my_total], device=device, dtype=torch.float64)
        t_max = torch.tensor([my_total], device=device, dtype=torch.float64)
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)

        if rank == 0:
            total_images = int(t_count.item())
            max_time = float(t_max.item())
            # Throughput dominated by max_time in parallel (makespan)
            img_per_min = total_images / (max_time / 60.0) if max_time > 0 else 0.0
            logger.info(f"{'='*70}")
            logger.info("Inference Complete (IMAGE-SHARDED + MAX CONFIDENCE)")
            logger.info(f"{'='*70}")
            logger.info(f"  total images processed: {total_images}")
            logger.info(f"  makespan (max rank time): {max_time:.1f}s ({max_time/60:.1f} min)")
            logger.info(f"  throughput (makespan): {img_per_min:.2f} images/min")
            logger.info(f"  output: {output_dir}")
            logger.info(f"{'='*70}")
    else:
        # Single-rank summary
        if rank == 0:
            total_images = processed_count
            max_time = my_total
            img_per_min = total_images / (max_time / 60.0) if max_time > 0 else 0.0
            logger.info(f"{'='*70}")
            logger.info("Inference Complete (single rank + MAX CONFIDENCE)")
            logger.info(f"{'='*70}")
            logger.info(f"  images processed: {total_images}")
            logger.info(f"  total time: {max_time:.1f}s ({max_time/60:.1f} min)")
            logger.info(f"  throughput: {img_per_min:.2f} images/min")
            logger.info(f"  output: {output_dir}")
            logger.info(f"{'='*70}")

    cleanup_distributed()


if __name__ == "__main__":
    main()