"""
SAM3 Distributed Segmentation Inference - Native Batching Version

This script performs distributed multi-GPU inference using SAM3's native batching API
for semantic segmentation of plant tissue images. Key optimizations:

1. TRUE BATCH PROCESSING: Multiple patches processed in a single GPU forward pass
2. MULTI-PROMPT PER PASS: All prompts for an image processed together, not sequentially
3. NATIVE SAM3 API: Uses datapoint/collate pattern from SAM3's official examples

This should be significantly faster than the processor-based approach.

Usage:
    torchrun --nproc_per_node=4 --nnodes=2 -m src.inference_v3_native \
        --input-dir /path/to/images \
        --output-dir /path/to/output \
        --prompts "cortex" "phloem" "xylem"

Author: ALS Photon Science Computing and Claude Opus 4.5
"""

import os
import sys
from pathlib import Path

# =============================================================================
# Environment Setup (must happen before other imports)
# =============================================================================
from dotenv import load_dotenv
load_dotenv()

if 'HF_HUB_CACHE' in os.environ:
    os.environ['HF_HOME'] = os.environ['HF_HUB_CACHE']

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

# SAM3 native imports
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata, 
    FindQueryLoaded, 
    Image as SAMImage, 
    Datapoint
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, 
    RandomResizeAPI, 
    ToTensorAPI, 
    NormalizeAPI
)
from sam3.eval.postprocessors import PostProcessImage

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global counter for unique query IDs
GLOBAL_COUNTER = 1


# =============================================================================
# SAM3 Native API Helper Functions (from official notebook)
# =============================================================================

def create_empty_datapoint() -> Datapoint:
    """Create an empty datapoint for a single image."""
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint: Datapoint, pil_image: Image.Image):
    """Add the image to be processed to the datapoint."""
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint: Datapoint, text_query: str) -> int:
    """
    Add a text query to the datapoint.
    
    Args:
        datapoint: The datapoint to add the query to
        text_query: The text prompt for segmentation
        
    Returns:
        Unique ID for this query (used to retrieve results)
    """
    global GLOBAL_COUNTER
    
    assert len(datapoint.images) == 1, "Please set the image first"
    
    h, w = datapoint.images[0].size  # Note: size is [h, w]
    
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],  # unused for inference
            is_exhaustive=True,    # unused for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1


# =============================================================================
# Image Processing Utilities
# =============================================================================

def load_state_dict_flexible(ckpt_path: str) -> dict:
    """Load a checkpoint and return a clean state_dict."""
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


def tif_to_rgb_pil(image_path: str, percentile_low: float = 0.5,
                   percentile_high: float = 99.5) -> Image.Image:
    """Convert a TIFF image to RGB PIL Image with proper normalization."""
    img = Image.open(image_path)
    
    if img.mode not in ['RGB', 'L', 'RGBA']:
        if img.mode in ['I', 'I;16', 'F']:
            img_array = np.array(img).astype(np.float32)
            p_low = np.percentile(img_array, percentile_low)
            p_high = np.percentile(img_array, percentile_high)
            img_array = np.clip(img_array, p_low, p_high)
            img_array = ((img_array - p_low) / (p_high - p_low) * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        else:
            img = img.convert('L')
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    return img


def crop_image_to_patches(image_pil: Image.Image, patch_size: int = 512
                          ) -> Tuple[List[Tuple], int, int, int, int]:
    """Crop a PIL image into a grid of patches."""
    W, H = image_pil.size
    patches = []
    
    n_rows = (H + patch_size - 1) // patch_size
    n_cols = (W + patch_size - 1) // patch_size
    
    for row in range(n_rows):
        for col in range(n_cols):
            y = row * patch_size
            x = col * patch_size
            
            patch_w = min(patch_size, W - x)
            patch_h = min(patch_size, H - y)
            
            patch = image_pil.crop((x, y, x + patch_w, y + patch_h))
            
            # Pad undersized patches
            if patch_w < patch_size or patch_h < patch_size:
                padded = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
                padded.paste(patch, (0, 0))
                patch = padded
            
            patches.append((patch, row, col, y, x, patch_h, patch_w))
    
    return patches, H, W, n_rows, n_cols


def load_images_from_dir(input_dir: str) -> List[Tuple[Image.Image, str]]:
    """Load all TIFF images from a directory."""
    input_path = Path(input_dir)
    tiff_files = sorted(
        list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
    )
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
    
    logger.info(f"Found {len(tiff_files)} TIFF files")
    
    images = []
    for tiff_file in tiff_files:
        img = tif_to_rgb_pil(str(tiff_file))
        images.append((img, tiff_file.name))
    
    logger.info(f"Loaded {len(images)} images")
    return images


# =============================================================================
# Dataset for Distributed Processing
# =============================================================================

class PatchDataset(Dataset):
    """
    Dataset that yields patch metadata.
    The actual datapoint creation happens in the processing function.
    """
    
    def __init__(self, patches: List[Tuple]):
        self.patches = patches
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[int, Tuple]:
        """Return index and patch data."""
        return idx, self.patches[idx]


# =============================================================================
# Distributed Setup
# =============================================================================

def setup_distributed() -> Tuple[int, int, int]:
    """Initialize PyTorch distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        if rank == 0:
            logger.info(f"Distributed initialized: {world_size} GPUs")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        logger.info("Running in single-GPU mode")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_results_parallel(
    local_results: List[dict],
    world_size: int,
    rank: int
) -> Optional[List[dict]]:
    """Gather results from all ranks to rank 0."""
    if world_size == 1:
        return local_results
    
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    
    if rank == 0:
        all_results = []
        for rank_results in gathered:
            all_results.extend(rank_results)
        all_results.sort(key=lambda x: x['idx'])
        logger.info(f"Gathered {len(all_results)} results from {world_size} GPUs")
        return all_results
    
    return None


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    bpe_path: str,
    device: torch.device,
    finetuned_checkpoint: str = None,
    original_checkpoint: str = None,
    rank: int = 0
) -> torch.nn.Module:
    """Load SAM3 model with optional finetuned weights."""
    if rank == 0:
        logger.info("Loading SAM3 model...")
    
    # Build model
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        enable_segmentation=True
    )
    
    if finetuned_checkpoint:
        if not original_checkpoint:
            raise ValueError("original_checkpoint required when using finetuned model")
        
        if rank == 0:
            logger.info(f"  Base checkpoint: {original_checkpoint}")
            logger.info(f"  Finetuned checkpoint: {finetuned_checkpoint}")
        
        # Load original weights first
        original_state = load_state_dict_flexible(original_checkpoint)
        model.load_state_dict(original_state, strict=False)
        
        # Overlay finetuned weights
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


def create_transform(image_size: int = 1008):
    """Create the SAM3 preprocessing transform."""
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=image_size, max_size=image_size, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_postprocessor(confidence_threshold: float = 0.5):
    """Create the SAM3 postprocessor."""
    return PostProcessImage(
        max_dets_per_img=-1,           # No limit, filter by confidence instead
        iou_type="segm",               # We want masks
        use_original_sizes_box=True,   # Resize boxes to original image size
        use_original_sizes_mask=True,  # Resize masks to original image size
        convert_mask_to_rle=False,     # Keep binary format for easy processing
        detection_threshold=confidence_threshold,
        to_cpu=False,
    )


# =============================================================================
# Core Inference with Native Batching
# =============================================================================

def process_patches_batch_native(
    patch_data_list: List[Tuple[int, Tuple]],
    prompts: List[str],
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    device: torch.device,
    patch_size: int
) -> List[dict]:
    """
    Process a batch of patches using SAM3's native batching API.
    
    KEY OPTIMIZATION: All patches AND all prompts are processed in a SINGLE
    forward pass through the model. This is much faster than:
    - Processing patches one at a time
    - Processing prompts sequentially
    
    Args:
        patch_data_list: List of (global_idx, patch_tuple) pairs
        prompts: List of text prompts
        model: SAM3 model
        transform: Preprocessing transform
        postprocessor: Results postprocessor
        device: Compute device
        patch_size: Size of patches
        
    Returns:
        List of result dicts with masks for each patch and prompt
    """
    global GLOBAL_COUNTER
    
    # Reset global counter for this batch to track query IDs
    batch_start_id = GLOBAL_COUNTER
    
    # ==========================================================================
    # Step 1: Create datapoints with ALL prompts for each patch
    # ==========================================================================
    datapoints = []
    query_id_map = {}  # Maps query_id -> (patch_idx_in_batch, prompt)
    
    for batch_idx, (global_idx, patch_tuple) in enumerate(patch_data_list):
        patch_pil, row, col, y_start, x_start, patch_h, patch_w = patch_tuple
        
        # Create datapoint for this patch
        dp = create_empty_datapoint()
        set_image(dp, patch_pil)
        
        # Add ALL prompts to this single datapoint
        for prompt in prompts:
            query_id = add_text_prompt(dp, prompt)
            query_id_map[query_id] = (batch_idx, prompt, global_idx, row, col, 
                                       y_start, x_start, patch_h, patch_w)
        
        # Apply transform
        dp = transform(dp)
        datapoints.append(dp)
    
    # ==========================================================================
    # Step 2: Collate into batch and move to GPU
    # ==========================================================================
    batch = collate(datapoints, dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, device, non_blocking=True)
    
    # ==========================================================================
    # Step 3: Single forward pass for ALL patches and ALL prompts
    # ==========================================================================
    with torch.inference_mode():
        output = model(batch)
    
    # ==========================================================================
    # Step 4: Postprocess results
    # ==========================================================================
    processed_results = postprocessor.process_results(output, batch.find_metadatas)
    
    # ==========================================================================
    # Step 5: Organize results by patch
    # ==========================================================================
    # Initialize results structure for each patch in batch
    batch_results = {}
    for batch_idx, (global_idx, patch_tuple) in enumerate(patch_data_list):
        patch_pil, row, col, y_start, x_start, patch_h, patch_w = patch_tuple
        batch_results[batch_idx] = {
            'idx': global_idx,
            'row': row,
            'col': col,
            'y_start': y_start,
            'x_start': x_start,
            'patch_h': patch_h,
            'patch_w': patch_w,
            'prompt_masks': {},
            'prompt_scores': {}
        }
    
    # Map results back to patches and prompts
    for query_id, (batch_idx, prompt, global_idx, row, col, 
                   y_start, x_start, patch_h, patch_w) in query_id_map.items():
        
        if query_id in processed_results:
            result = processed_results[query_id]
            
            # Extract masks and scores
            if 'masks' in result and len(result['masks']) > 0:
                masks = result['masks']
                scores = result['scores'] if 'scores' in result else None
                
                # Convert to numpy
                if torch.is_tensor(masks):
                    masks = masks.cpu().numpy()
                if scores is not None and torch.is_tensor(scores):
                    scores = scores.cpu().numpy()
                
                batch_results[batch_idx]['prompt_masks'][prompt] = masks
                batch_results[batch_idx]['prompt_scores'][prompt] = scores
            else:
                batch_results[batch_idx]['prompt_masks'][prompt] = None
                batch_results[batch_idx]['prompt_scores'][prompt] = None
        else:
            batch_results[batch_idx]['prompt_masks'][prompt] = None
            batch_results[batch_idx]['prompt_scores'][prompt] = None
    
    return list(batch_results.values())


def process_patches_distributed_native(
    patches: List[Tuple],
    prompts: List[str],
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    patch_size: int
) -> List[dict]:
    """
    Process patches across multiple GPUs using native SAM3 batching.
    
    Args:
        patches: List of all patches
        prompts: List of text prompts
        model: SAM3 model
        transform: Preprocessing transform
        postprocessor: Results postprocessor
        batch_size: Number of patches per batch
        rank: Process rank
        world_size: Total processes
        device: Compute device
        patch_size: Size of patches
        
    Returns:
        List of result dicts for patches processed by this rank
    """
    # Create dataset
    dataset = PatchDataset(patches)
    
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = None
    
    # Simple dataloader - just yields indices and patch tuples
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # Keep simple for now
        shuffle=False if sampler else False,
        collate_fn=lambda x: x  # Don't collate, we'll do it ourselves
    )
    
    if rank == 0:
        logger.info(f"Processing {len(dataset)} patches with batch_size={batch_size}")
        logger.info(f"  Batches per GPU: {len(dataloader)}")
        logger.info(f"  Prompts per patch: {len(prompts)}")
        logger.info(f"  Total forward passes: {len(dataloader)} (vs {len(dataloader) * len(prompts)} without multi-prompt)")
    
    local_results = []
    
    for batch_idx, batch_data in enumerate(dataloader):
        # batch_data is a list of (global_idx, patch_tuple) pairs
        batch_results = process_patches_batch_native(
            patch_data_list=batch_data,
            prompts=prompts,
            model=model,
            transform=transform,
            postprocessor=postprocessor,
            device=device,
            patch_size=patch_size
        )
        local_results.extend(batch_results)
        
        if rank == 0 and (batch_idx + 1) % 5 == 0:
            logger.info(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    if rank == 0:
        logger.info(f"  Rank 0 processed {len(local_results)} patches")
    
    return local_results


# =============================================================================
# Post-Processing: Stitching and Saving
# =============================================================================

def stitch_masks_for_prompt(
    results: List[dict],
    prompt: str,
    orig_h: int,
    orig_w: int,
    patch_size: int
) -> np.ndarray:
    """Stitch patch masks back into a full-image mask for one prompt."""
    stitched = np.zeros((orig_h, orig_w), dtype=np.float32)
    
    for result in results:
        masks = result['prompt_masks'].get(prompt)
        y_start = result['y_start']
        x_start = result['x_start']
        patch_h = result['patch_h']
        patch_w = result['patch_w']
        
        if masks is None or (isinstance(masks, np.ndarray) and masks.size == 0):
            continue
        
        # Combine multiple masks in this patch
        combined = np.zeros((patch_size, patch_size), dtype=np.float32)
        
        masks = np.array(masks)
        if masks.ndim == 2:
            combined = masks
        elif masks.ndim == 3:
            combined = masks.max(axis=0)
        elif masks.ndim == 4:
            combined = masks.squeeze(1).max(axis=0)
        
        # Resize if needed
        if combined.shape != (patch_size, patch_size):
            m_tensor = torch.from_numpy(combined).float()[None, None, ...]
            m_tensor = torch.nn.functional.interpolate(
                m_tensor, size=(patch_size, patch_size),
                mode="bilinear", align_corners=False
            )
            combined = m_tensor.squeeze().numpy()
        
        # Crop to actual patch dimensions
        combined_crop = combined[:patch_h, :patch_w]
        
        # Place in stitched output
        stitched[y_start:y_start+patch_h, x_start:x_start+patch_w] = combined_crop
    
    return stitched


def save_mask(mask: np.ndarray, output_path: Path):
    """Save a mask as a TIFF file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if mask.max() <= 1.0:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)
    
    Image.fromarray(mask_uint8).save(output_path, compression="tiff_deflate")


def save_combined_mask(
    all_masks: Dict[str, np.ndarray],
    class_colors: Dict[str, int],
    output_path: Path
):
    """Create and save a combined segmentation mask with class IDs."""
    first_prompt = list(all_masks.keys())[0]
    H, W = all_masks[first_prompt].shape
    
    combined = np.zeros((H, W), dtype=np.uint8)
    
    for prompt, class_id in class_colors.items():
        if prompt in all_masks:
            mask = all_masks[prompt]
            binary = mask > 0.5
            combined[binary] = class_id
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(output_path, compression="tiff_deflate")


def process_and_save_image(
    image_pil: Image.Image,
    image_name: str,
    model: torch.nn.Module,
    transform: ComposeAPI,
    postprocessor: PostProcessImage,
    prompts: List[str],
    class_colors: Dict[str, int],
    output_dir: Path,
    patch_size: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device
):
    """Process a single image and save results immediately."""
    global GLOBAL_COUNTER
    GLOBAL_COUNTER = 1  # Reset counter for each image
    
    # Create patches
    patches, orig_h, orig_w, n_rows, n_cols = crop_image_to_patches(
        image_pil, patch_size
    )
    
    if rank == 0:
        logger.info(f"  Image size: {orig_w}x{orig_h}")
        logger.info(f"  Patches: {len(patches)} ({n_rows}x{n_cols})")
    
    # Process all patches with all prompts
    start_time = time.time()
    local_results = process_patches_distributed_native(
        patches=patches,
        prompts=prompts,
        model=model,
        transform=transform,
        postprocessor=postprocessor,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
        patch_size=patch_size
    )
    
    # Gather results from all GPUs
    all_results = gather_results_parallel(local_results, world_size, rank)
    
    # Only rank 0 does stitching and saving
    if rank == 0:
        inference_time = time.time() - start_time
        logger.info(f"  Inference completed in {inference_time:.1f}s")
        
        # Stitch and save each prompt's mask
        all_stitched = {}
        image_stem = Path(image_name).stem
        
        for prompt in prompts:
            stitched = stitch_masks_for_prompt(
                all_results, prompt, orig_h, orig_w, patch_size
            )
            all_stitched[prompt] = stitched
            
            # Save individual prompt mask
            prompt_dir = output_dir / prompt.replace(" ", "_")
            save_mask(stitched, prompt_dir / f"{image_stem}_mask.tif")
        
        # Save combined multi-class mask
        combined_path = output_dir / "combined" / f"{image_stem}_combined.tif"
        save_combined_mask(all_stitched, class_colors, combined_path)
        
        logger.info(f"  ✓ Saved masks for {image_name}")
    
    # Synchronize before next image
    if world_size > 1:
        dist.barrier()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for distributed SAM3 inference with native batching."""
    
    parser = argparse.ArgumentParser(
        description="SAM3 Distributed Inference with Native Batching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--bpe-path", type=str, required=True)
    
    # Optional paths
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--finetuned-checkpoint", type=str, default=None)
    parser.add_argument("--original-checkpoint", type=str, default=None)
    
    # Processing parameters
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Patches per batch - can be larger with native batching")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=1008,
                        help="Size for SAM3 transform (default from official examples)")
    
    # Prompts
    parser.add_argument("--prompts", nargs="+",
                        default=["cortex", "Phloem Fibers", "Xylem vessels", 
                                 "Pith cells", "outer cells"])
    
    # Resume capability
    parser.add_argument("--skip-existing", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    if world_size > 1:
        dist.barrier()
    
    # Load images
    images = load_images_from_dir(args.input_dir)
    
    # Skip existing if requested
    if args.skip_existing:
        combined_dir = output_dir / "combined"
        existing = set(f.stem.replace("_combined", "") for f in combined_dir.glob("*.tif"))
        original_count = len(images)
        images = [(img, name) for img, name in images 
                  if Path(name).stem not in existing]
        if rank == 0:
            logger.info(f"Skipping {original_count - len(images)} already-processed images")
            logger.info(f"Remaining: {len(images)} images")
    
    if not images:
        if rank == 0:
            logger.info("No images to process. Exiting.")
        cleanup_distributed()
        return
    
    # Load model
    model = load_model(
        bpe_path=args.bpe_path,
        device=device,
        finetuned_checkpoint=args.finetuned_checkpoint,
        original_checkpoint=args.original_checkpoint,
        rank=rank
    )
    
    # Note: We don't wrap with DDP since we're using native SAM3 batching
    # which handles the model internally
    
    # Create transform and postprocessor
    transform = create_transform(args.image_size)
    postprocessor = create_postprocessor(args.confidence)
    
    # Process images
    class_colors = {prompt: idx + 1 for idx, prompt in enumerate(args.prompts)}
    
    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info("Starting distributed inference (Native Batching)")
        logger.info(f"{'='*60}")
        logger.info(f"  GPUs: {world_size}")
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Prompts: {args.prompts}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  All prompts processed per forward pass: YES")
        logger.info(f"  Class mapping: {class_colors}")
        logger.info(f"{'='*60}\n")
    
    total_start = time.time()
    
    for img_idx, (image_pil, image_name) in enumerate(images):
        if rank == 0:
            logger.info(f"\n[{img_idx+1}/{len(images)}] Processing: {image_name}")
        
        process_and_save_image(
            image_pil=image_pil,
            image_name=image_name,
            model=model,
            transform=transform,
            postprocessor=postprocessor,
            prompts=args.prompts,
            class_colors=class_colors,
            output_dir=output_dir,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            device=device
        )
    
    # Summary
    if rank == 0:
        total_time = time.time() - total_start
        images_per_min = len(images) / (total_time / 60)
        
        logger.info(f"\n{'='*60}")
        logger.info("Inference Complete")
        logger.info(f"{'='*60}")
        logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"  Images processed: {len(images)}")
        logger.info(f"  Throughput: {images_per_min:.1f} images/min")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"{'='*60}\n")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
