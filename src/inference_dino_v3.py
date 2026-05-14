import lightly_train
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os

# ------------------------
# Project definitions
# ------------------------
PROJECTS = {
    "synaps-I": {
        "colors": {
            0: [128, 128, 128],
            1: [0,   0,   255],
            2: [0,   255,   0],
            3: [128,   0, 128],
            4: [255,   0,   0],
            5: [255, 255,   0],
            6: [255, 165,   0],
        },
        "names": {
            0: "background",
            1: "Cortex",
            2: "Phloem Fibers",
            3: "Phloem",
            4: "Xylem vessels",
            5: "Air-based Pith cells",
            6: "Water-based Pith cells",
        },
    },
    "moon": {
        "colors": {
            0: [128, 128, 128],
            1: [0,   120, 255],
            2: [255,  80,  80],
            3: [ 50, 200,  50],
        },
        "names": {
            0: "background",
            1: "ice",
            2: "particles",
            3: "pores",
        },
    },
    "leaf": {
        "colors": {
            170: [170, 170, 170],
            85:  [85,  85,  85],
            0:   [0,   0,   0],
            255: [255, 255, 255],
            180: [180, 180, 180],
        },
        # Distinct bright colors used only for overlay visualization
        "overlay_colors": {
            170: [128, 128, 128],   # Gray        - background
            85:  [0,   255,   0],   # Green       - epidermis
            0:   [0,   0,   255],   # Blue        - mesophyll
            255: [255, 255,   0],   # Yellow      - air_space
            180: [255,   0,   0],   # Red         - vein
        },
        "names": {
            170: "background",
            85:  "epidermis",
            0:   "mesophyll",
            255: "air_space",
            180: "vein",
        },
    },
}

IGNORE_CLASSES = [255]

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

def mask_to_color(mask_array, class_colors):
    """
    Convert a grayscale mask to RGB using class colors.
    
    Args:
        mask_array: 2D numpy array with class indices
        class_colors: dict mapping class id -> [R, G, B]
    
    Returns:
        RGB image as numpy array
    """
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        color_mask[mask_array == class_id] = color
    
    return color_mask

def normalize_image_percentile(img_array, lower_percentile=2, upper_percentile=98):
    """
    Normalize image using percentiles for high contrast display.
    
    Args:
        img_array: numpy array (H, W, 3) or (H, W)
        lower_percentile: lower percentile to clip (default 2)
        upper_percentile: upper percentile to clip (default 98)
    
    Returns:
        Normalized image in range [0, 255] as uint8
    """
    lower = np.percentile(img_array, lower_percentile)
    upper = np.percentile(img_array, upper_percentile)
    
    img_clipped = np.clip(img_array, lower, upper)
    
    if upper > lower:
        img_normalized = ((img_clipped - lower) / (upper - lower) * 255).astype(np.uint8)
    else:
        img_normalized = img_array.astype(np.uint8)
    
    return img_normalized

# ------------------------
# Custom collate: keep images as lists to support variable sizes
# ------------------------
def custom_collate(batch):
    """
    Keep images as lists instead of stacking into a tensor.
    predict_single() is called per image, so no padding or stacking needed.
    """
    return {
        'image':    [item['image']    for item in batch],
        'image_np': [item['image_np'] for item in batch],
        'img_path': [item['img_path'] for item in batch],
        'img_name': [item['img_name'] for item in batch],
    }


# ------------------------
# Dataset class
# ------------------------
class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = Path(img_dir)
        
        IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        self.img_paths = sorted([
            p for p in self.img_dir.iterdir() 
            if p.suffix.lower() in IMG_EXTS
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        with Image.open(img_path) as img:
            if img.mode not in ['RGB', 'L', 'RGBA']:
                if img.mode in ['I', 'I;16', 'F']:
                    img_array = np.array(img)
                    img_array = ((img_array - img_array.min()) / 
                                (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='L')
                else:
                    img = img.convert('L')
            
            img = img.convert("RGB")
            img_np = np.array(img)
        
        img_tensor = transforms.ToTensor()(img)  # [0, 1] range
        
        return {
            'image': img_tensor,
            'image_np': img_np,
            'img_path': str(img_path),
            'img_name': img_path.name
        }

# ------------------------
# Batch prediction function
# ------------------------
def predict_batch(model, images):
    """
    Run inference on a batch of same-size images in a single forward pass.

    Args:
        model:  The loaded model
        images: List of (C, H, W) tensors — all must have identical spatial dims.

    Returns:
        List of predicted masks as numpy arrays, one per input image.
    """
    if model.training:
        model.eval()

    device = next(model.parameters()).device

    # Stack into (B, C, H, W) — safe because all images are the same size
    batch = torch.stack(images).to(device)          # (B, C, H, W)
    B, C, image_h, image_w = batch.shape

    if batch.dtype != torch.float32:
        batch = batch.to(dtype=torch.float32)

    # Normalize entire batch at once
    mean = torch.tensor(model.image_normalize["mean"], device=device).view(1, 3, 1, 1)
    std  = torch.tensor(model.image_normalize["std"],  device=device).view(1, 3, 1, 1)
    batch = (batch - mean) / std

    # Resize short side to crop_size (preserves aspect ratio)
    crop_size = min(model.image_size)
    scale = crop_size / min(image_h, image_w)
    new_h, new_w = int(image_h * scale), int(image_w * scale)
    batch = F.interpolate(batch, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # Single forward pass for the whole batch
    with torch.inference_mode():
        logits = model._forward_logits(batch)   # (B, K+1, H', W')
    logits = logits[:, :-1]                     # (B, K,   H', W')

    # Resize predictions back to original spatial size
    logits = F.interpolate(logits, size=(image_h, image_w), mode="bilinear", align_corners=False)

    masks = logits.argmax(dim=1)                        # (B, H, W)
    masks = model.internal_class_to_class[masks]        # (B, H, W)

    return [masks[i].cpu().numpy() for i in range(B)]

# ------------------------
# Main function
# ------------------------
def main():
    parser = argparse.ArgumentParser(description='Distributed DINO inference')
    parser.add_argument('--input-dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--finetuned-checkpoint', type=str, required=True, help='Path to finetuned checkpoint')
    parser.add_argument('--save-overlay', action='store_true', help='Save overlay images')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--project', type=str, default='synaps-I',
                        choices=list(PROJECTS.keys()),
                        help='Project name (determines class colors and names)')
    args = parser.parse_args()

    # Resolve project-specific class definitions
    CLASS_COLORS         = PROJECTS[args.project]["colors"]
    CLASS_OVERLAY_COLORS = PROJECTS[args.project].get("overlay_colors", CLASS_COLORS)
    CLASS_NAMES          = PROJECTS[args.project]["names"]

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Setup paths
    IMG_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_MASK_DIR = OUTPUT_DIR / "semantic_masks"
    OUTPUT_OVERLAY_DIR = OUTPUT_DIR / "semantic_masks_overlay"
    CKPT_PATH = args.finetuned_checkpoint
    
    # Create output directories (only on rank 0)
    if rank == 0:
        OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
        if args.save_overlay:
            OUTPUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    
    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        print(f"Project: {args.project}")
        print(f"Classes: {CLASS_NAMES}")
        print("Loading model...")
    model = lightly_train.load_model(CKPT_PATH)
    model.eval()
    model = model.to(device)
    if rank == 0:
        print(f"Model loaded on {world_size} GPU(s)")
    
    dataset = ImageDataset(IMG_DIR)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )
    
    if rank == 0:
        print(f"Total images: {len(dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Number of GPUs: {world_size}")
        print(f"Images per GPU: {len(dataset) // world_size}")
        print("-" * 70)
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            # --- bucket by (H, W) so same-size images share one forward pass ---
            buckets = {}
            for i, img in enumerate(batch['image']):
                key = (img.shape[1], img.shape[2])   # (H, W)
                buckets.setdefault(key, []).append(i)

            pred_mask_map = {}   # index -> pred_mask_np
            for size_key, indices in buckets.items():
                imgs = [batch['image'][i] for i in indices]
                masks = predict_batch(model, imgs)
                for i, mask in zip(indices, masks):
                    pred_mask_map[i] = mask
            # -------------------------------------------------------------------

            for i in range(len(batch['img_name'])):
                pred_mask_np = pred_mask_map[i]

                img_np = batch['image_np'][i]
                if isinstance(img_np, torch.Tensor):
                    img_np = img_np.cpu().numpy()

                img_name = batch['img_name'][i]

                pred_unique = np.unique(pred_mask_np)

                if rank == 0:
                    print(f"GPU {rank}: {img_name}")
                    print(f"  Predicted classes: {pred_unique}")

                mask_save_path = OUTPUT_MASK_DIR / img_name
                Image.fromarray(pred_mask_np.astype(np.uint8), mode='L').save(mask_save_path)

                if rank == 0:
                    print(f"  Saved mask to: {mask_save_path}")

                if args.save_overlay:
                    pred_color_mask = mask_to_color(pred_mask_np, CLASS_OVERLAY_COLORS)
                    img_np_normalized = normalize_image_percentile(img_np, 2, 98)
                    pred_overlay = (0.6 * img_np_normalized + 0.4 * pred_color_mask).astype(np.uint8)

                    overlay_save_path = OUTPUT_OVERLAY_DIR / img_name
                    Image.fromarray(pred_overlay).save(overlay_save_path)

                    if rank == 0:
                        print(f"  Saved overlay to: {overlay_save_path}")

                if rank == 0:
                    print("-" * 70)
    
    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        print(f"\nInference complete!")
        print(f"Masks saved to: {OUTPUT_MASK_DIR}")
        if args.save_overlay:
            print(f"Overlays saved to: {OUTPUT_OVERLAY_DIR}")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()