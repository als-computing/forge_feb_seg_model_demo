from cellpose import models
from glob import glob
from natsort import natsorted
import tifffile
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
import torch
import torch.distributed as dist
import os

def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Cleanup distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def normalize_image_percentile(image, lower_percentile=1, upper_percentile=99):
    """
    Normalize image using percentile-based scaling for better contrast.
    
    Args:
        image: Input image array
        lower_percentile: Lower percentile for clipping (default: 1)
        upper_percentile: Upper percentile for clipping (default: 99)
    
    Returns:
        Normalized image in range [0, 255] as uint8
    """
    p_low = np.percentile(image, lower_percentile)
    p_high = np.percentile(image, upper_percentile)
    
    img_clipped = np.clip(image, p_low, p_high)
    img_normalized = ((img_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    
    return img_normalized

def main():
    parser = argparse.ArgumentParser(description='Distributed Cellpose Inference')
    parser.add_argument('--input-dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--finetuned-checkpoint', type=str, required=True, help='Path to finetuned model')
    parser.add_argument('--save-overlay', action='store_true', help='Save overlay images (default: False)')
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Define paths
    image_dir = args.input_dir
    model_path = args.finetuned_checkpoint
    
    # Create output directories (only rank 0)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "instance_masks"
    
    if rank == 0:
        mask_dir.mkdir(parents=True, exist_ok=True)
        if args.save_overlay:
            overlay_dir = output_dir / "instance_masks_overlay"
            overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()
    
    # Load all TIFF files
    if rank == 0:
        print("Loading images...")
    
    test_files = natsorted(glob(f"{image_dir}/*.tif") + glob(f"{image_dir}/*.tiff"))
    test_files = [f for f in test_files if "_masks" not in f and "_flows" not in f]
    
    # Distribute files across GPUs
    files_per_gpu = len(test_files) // world_size
    remainder = len(test_files) % world_size
    
    if rank < remainder:
        start_idx = rank * (files_per_gpu + 1)
        end_idx = start_idx + files_per_gpu + 1
    else:
        start_idx = rank * files_per_gpu + remainder
        end_idx = start_idx + files_per_gpu
    
    local_files = test_files[start_idx:end_idx]
    local_data = [tifffile.imread(f) for f in local_files]
    
    if rank == 0:
        print(f"Total images: {len(test_files)}")
    print(f"Rank {rank}: Processing {len(local_data)} images")
    
    # Load model and run inference
    if rank == 0:
        print("Running inference...")
    
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    masks = model.eval(local_data, batch_size=32)[0]
    
    if rank == 0:
        print(f"Inference complete! Detected cells in each image:")
    
    # Save masks and overlays
    for i, (file, mask, image) in enumerate(zip(local_files, masks, local_data)):
        filename = Path(file).stem
        print(f"Rank {rank}: {Path(file).name}: {mask.max()} cells")
        
        # Save instance mask as TIFF
        tifffile.imwrite(mask_dir / f"{filename}_mask.tif", mask.astype(np.uint16))
        
        # Create and save overlay if requested
        if args.save_overlay:
            # Create overlay using the new method
            mask_np = np.array(mask)
            img_np = image
            
            # If mask is RGB, take one channel
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            
            # Normalize image for display using percentile normalization
            img_disp = normalize_image_percentile(img_np, lower_percentile=1, upper_percentile=99)
            
            # Pre-compute colors for ALL objects (fixed seed for consistency)
            rng = np.random.default_rng(42)  # Fixed seed
            all_instance_ids = np.unique(mask_np)
            color_map = {}
            for instance_id in all_instance_ids:
                if instance_id == 0:  # Skip background
                    continue
                color_map[instance_id] = rng.random(3)
            
            # Create full image overlay
            rgb_mask_full = np.zeros((*mask_np.shape, 3), dtype=np.float32)
            for instance_id in np.unique(mask_np):
                if instance_id == 0:
                    continue
                rgb_mask_full[mask_np == instance_id] = color_map[instance_id]
            
            # Create full overlay
            overlay_full = np.stack([img_disp, img_disp, img_disp], axis=-1).astype(np.float32) / 255.0
            overlay_full = overlay_full * 0.8 + rgb_mask_full * 0.2
            overlay_full = (overlay_full * 255).astype(np.uint8)
            
            # Save overlay
            overlay_dir = output_dir / "instance_masks_overlay"
            overlay_img = Image.fromarray(overlay_full)
            overlay_img.save(overlay_dir / f"{filename}_overlay.png")
    
    # Synchronize all processes before final print
    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        print(f"\nSaved {len(test_files)} instance masks to {mask_dir}")
        if args.save_overlay:
            print(f"Saved {len(test_files)} overlays to {output_dir / 'instance_masks_overlay'}")
    
    # Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    main()