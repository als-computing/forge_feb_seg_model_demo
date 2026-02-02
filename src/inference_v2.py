import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if 'HF_HUB_CACHE' in os.environ:
    os.environ['HF_HOME'] = os.environ['HF_HUB_CACHE']

import argparse
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Utils (from your working code)
# =============================================================================
def load_state_dict_flexible(ckpt_path: str) -> dict:
    """Load a checkpoint and return a clean state_dict (no module. prefix)."""
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


def tif_to_rgb_pil(image_path: str) -> Image.Image:
    """Convert TIFF to RGB PIL Image."""
    img = Image.open(image_path)
    
    if img.mode not in ['RGB', 'L', 'RGBA']:
        if img.mode in ['I', 'I;16', 'F']:
            img_array = np.array(img).astype(np.float32)
            p_low = np.percentile(img_array, 0.5)
            p_high = np.percentile(img_array, 99.5)
            img_array = np.clip(img_array, p_low, p_high)
            img_array = ((img_array - p_low) / (p_high - p_low) * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
        else:
            img = img.convert('L')
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    return img


def to_numpy(x):
    """Convert tensor to numpy safely."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return x


def extract_masks_scores(inference_state: dict):
    """Extract masks and scores from inference state."""
    masks = inference_state.get("masks", inference_state.get("pred_masks", None))
    scores = inference_state.get("scores", inference_state.get("pred_logits", None))

    masks = to_numpy(masks)
    scores = to_numpy(scores)

    # Normalize scores safely
    if scores is not None:
        scores = np.array(scores)
        if scores.size == 0:
            scores = None
        else:
            if scores.ndim > 1:
                scores = scores.max(axis=-1)
            if scores.size > 0 and (np.nanmax(scores) > 10 or np.nanmin(scores) < -10):
                scores = 1 / (1 + np.exp(-scores))

    # Normalize masks safely
    if masks is not None:
        masks = np.array(masks)
        if masks.size == 0:
            masks = None

    return masks, scores


def crop_image_to_patches(image_pil: Image.Image, patch_size: int = 512):
    """
    Crop a PIL image into patches.
    Returns list of (patch_pil, row, col, y_start, x_start, patch_h, patch_w) tuples.
    """
    W, H = image_pil.size
    patches = []
    
    n_rows = (H + patch_size - 1) // patch_size
    n_cols = (W + patch_size - 1) // patch_size
    
    for row in range(n_rows):
        for col in range(n_cols):
            y = row * patch_size
            x = col * patch_size
            
            # Calculate actual patch dimensions (handle edges)
            patch_w = min(patch_size, W - x)
            patch_h = min(patch_size, H - y)
            
            # Crop patch
            patch = image_pil.crop((x, y, x + patch_w, y + patch_h))
            
            # If patch is smaller than patch_size, pad it
            if patch_w < patch_size or patch_h < patch_size:
                padded = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
                padded.paste(patch, (0, 0))
                patch = padded
            
            patches.append((patch, row, col, y, x, patch_h, patch_w))
    
    return patches, H, W, n_rows, n_cols


def load_images_from_dir(input_dir: str) -> List[Tuple[Image.Image, str]]:
    """Load TIFF images from directory."""
    input_path = Path(input_dir)
    tiff_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
    
    logger.info(f"Loading {len(tiff_files)} images...")
    
    images = []
    for tiff_file in tiff_files:
        img = tif_to_rgb_pil(str(tiff_file))
        images.append((img, tiff_file.name))
    
    logger.info(f"Loaded {len(images)} images")
    return images


# =============================================================================
# Distributed Processing
# =============================================================================
class PatchDataset(Dataset):
    """Dataset for image patches."""
    
    def __init__(self, patches: List[Tuple]):
        self.patches = patches
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx], idx


def setup_distributed():
    """Initialize distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        if rank == 0:
            logger.info(f"Initialized distributed: {world_size} GPUs")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        logger.info("Running in single-GPU mode")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model(
    bpe_path: str,
    device: torch.device,
    finetuned_checkpoint: str = None,
    original_checkpoint: str = None,
    rank: int = 0
):
    """Load SAM3 model (finetuned or original)."""
    if rank == 0:
        logger.info("Loading SAM3 model...")
    
    if finetuned_checkpoint:
        if not original_checkpoint:
            raise ValueError("original_checkpoint required when using finetuned model")
        
        if rank == 0:
            logger.info(f"Loading finetuned model from: {finetuned_checkpoint}")
        
        # Load original state
        original_state = load_state_dict_flexible(original_checkpoint)
        
        # Build model
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            enable_segmentation=True
        )
        
        # Load original weights first
        _ = model.load_state_dict(original_state, strict=False)
        
        # Load finetuned weights
        finetuned_state = load_state_dict_flexible(finetuned_checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(finetuned_state, strict=False)
        
        if rank == 0:
            if missing_keys:
                logger.info(f"⚠ Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.info(f"⚠ Unexpected keys: {len(unexpected_keys)}")
            logger.info("✓ Finetuned model loaded")
    else:
        if rank == 0:
            logger.info("Loading original SAM3 model...")
        
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            enable_segmentation=True
        )
        
        if rank == 0:
            logger.info("✓ Original model loaded")
    
    model.eval().to(device)
    
    return model


def process_patches_distributed(
    patches: List[Tuple],
    processor,
    prompt: str,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device
):
    """Process patches in distributed manner without DataLoader."""
    
    # Manually distribute patches across ranks
    total_patches = len(patches)
    patches_per_rank = (total_patches + world_size - 1) // world_size
    
    start_idx = rank * patches_per_rank
    end_idx = min(start_idx + patches_per_rank, total_patches)
    
    my_patches = patches[start_idx:end_idx]
    
    if rank == 0:
        logger.info(f"Total patches: {total_patches}")
        logger.info(f"Rank {rank} processing patches {start_idx} to {end_idx-1} ({len(my_patches)} patches)")
    
    local_results = []
    
    # Process each patch assigned to this rank
    for local_idx, patch_data in enumerate(my_patches):
        global_idx = start_idx + local_idx
        
        # Unpack patch tuple
        patch_pil, row, col, y_start, x_start, patch_h, patch_w = patch_data
        
        # Run inference using set_image (like your working code)
        state = processor.set_image(patch_pil)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        
        masks, scores = extract_masks_scores(state)
        
        # Store results with metadata
        local_results.append({
            'masks': masks,
            'scores': scores,
            'row': row,
            'col': col,
            'y_start': y_start,
            'x_start': x_start,
            'patch_h': patch_h,
            'patch_w': patch_w,
            'idx': global_idx
        })
        
        # Progress logging
        if rank == 0 and (local_idx + 1) % 10 == 0:
            logger.info(f"  Processed {local_idx + 1}/{len(my_patches)} patches")
    
    return local_results


def gather_all_results(local_results: List[dict], world_size: int, rank: int):
    """Gather results from all ranks."""
    if world_size == 1:
        return local_results
    
    if rank == 0:
        logger.info("Gathering results from all GPUs...")
    
    # Gather all results to rank 0
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    
    if rank == 0:
        # Flatten and sort by index
        all_results = []
        for rank_results in gathered:
            all_results.extend(rank_results)
        
        all_results.sort(key=lambda x: x['idx'])
        
        logger.info(f"Gathered {len(all_results)} total results")
        return all_results
    else:
        return None


def stitch_patches(
    results: List[dict],
    orig_h: int,
    orig_w: int,
    patch_size: int
):
    """Stitch patches back into full image masks."""
    logger.info("Stitching patches back into full image...")
    
    # Create output mask
    stitched_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
    
    for result in results:
        masks = result['masks']
        y_start = result['y_start']
        x_start = result['x_start']
        patch_h = result['patch_h']
        patch_w = result['patch_w']
        
        if masks is None or len(masks) == 0:
            continue
        
        # Combine multiple masks in this patch
        combined_mask = np.zeros((patch_size, patch_size), dtype=np.float32)
        for m in masks:
            if m.ndim == 3:
                m = m.squeeze(0)
            
            # Resize if needed
            if m.shape != (patch_size, patch_size):
                m_tensor = torch.from_numpy(m).float()[None, None, ...]
                m_tensor = torch.nn.functional.interpolate(
                    m_tensor, size=(patch_size, patch_size), 
                    mode="bilinear", align_corners=False
                )
                m = m_tensor.squeeze().numpy()
            
            combined_mask = np.maximum(combined_mask, m)
        
        # Crop to actual patch size and place in stitched image
        mask_crop = combined_mask[:patch_h, :patch_w]
        stitched_mask[y_start:y_start+patch_h, x_start:x_start+patch_w] = mask_crop
    
    return stitched_mask


def save_results(mask: np.ndarray, prompt: str, image_name: str, output_dir: Path):
    """Save mask as TIFF."""
    prompt_dir = output_dir / prompt.replace(" ", "_")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to uint8
    if mask.max() <= 1.0:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)
    
    output_path = prompt_dir / f"{Path(image_name).stem}_mask.tif"
    Image.fromarray(mask_uint8).save(output_path, compression="tiff_deflate")
    
    logger.info(f"Saved mask to {output_path}")


def create_combined_mask(
    all_masks: Dict[str, np.ndarray],
    class_colors: Dict[str, int],
    image_name: str,
    output_dir: Path
):
    """Create combined segmentation mask with class IDs."""
    logger.info("Creating combined mask...")
    
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(exist_ok=True)
    
    # Get dimensions from first mask
    first_prompt = list(all_masks.keys())[0]
    H, W = all_masks[first_prompt].shape
    
    combined = np.zeros((H, W), dtype=np.uint8)
    
    for prompt, class_id in class_colors.items():
        mask = all_masks[prompt]
        binary_mask = mask > 0.5
        combined[binary_mask] = class_id
    
    output_path = combined_dir / f"{Path(image_name).stem}_combined.tif"
    Image.fromarray(combined).save(output_path, compression="tiff_deflate")
    
    logger.info(f"Saved combined mask to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 Distributed Inference")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--patch-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--prompts", nargs="+", default=["cells"])
    parser.add_argument("--bpe-path", type=str, required=True)
    parser.add_argument("--finetuned-checkpoint", type=str, default=None)
    parser.add_argument("--original-checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Synchronize before loading
    if world_size > 1:
        dist.barrier()
    
    # Load images (all ranks load independently - simpler and avoids broadcast issues)
    images = load_images_from_dir(args.input_dir)
    
    # Load model
    model = load_model(
        bpe_path=args.bpe_path,
        device=device,
        finetuned_checkpoint=args.finetuned_checkpoint,
        original_checkpoint=args.original_checkpoint,
        rank=rank
    )
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Setup processor
    actual_model = model.module if isinstance(model, DDP) else model
    processor = Sam3Processor(actual_model, confidence_threshold=args.confidence)
    
    # Process each image
    class_colors = {prompt: idx + 1 for idx, prompt in enumerate(args.prompts)}
    
    for image_pil, image_name in images:
        if rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {image_name}")
            logger.info(f"{'='*60}")
        
        # Create patches
        patches, orig_h, orig_w, n_rows, n_cols = crop_image_to_patches(
            image_pil, args.patch_size
        )
        
        if rank == 0:
            logger.info(f"Image size: {orig_w}x{orig_h}")
            logger.info(f"Patches: {len(patches)} ({n_rows}x{n_cols})")
        
        all_stitched_masks = {}
        
        # Process each prompt
        for prompt in args.prompts:
            if rank == 0:
                logger.info(f"\nProcessing prompt: '{prompt}'")
            
            start_time = time.time()
            
            # Process patches
            local_results = process_patches_distributed(
                patches, processor, prompt, args.batch_size,
                rank, world_size, device
            )
            
            # Gather results
            all_results = gather_all_results(local_results, world_size, rank)
            
            if rank == 0:
                # Stitch and save
                stitched_mask = stitch_patches(
                    all_results, orig_h, orig_w, args.patch_size
                )
                
                save_results(stitched_mask, prompt, image_name, output_dir)
                all_stitched_masks[prompt] = stitched_mask
                
                elapsed = time.time() - start_time
                logger.info(f"✓ Completed '{prompt}' in {elapsed:.1f}s")
        
        # Create combined mask
        if rank == 0:
            create_combined_mask(all_stitched_masks, class_colors, image_name, output_dir)
    
    if rank == 0:
        logger.info(f"\n✓ All results saved to {output_dir}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()