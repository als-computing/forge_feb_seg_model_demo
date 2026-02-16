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

# ------------------------
# Class definitions
# ------------------------
CLASS_COLORS = {
    0: [128,128,128],    # gray - background
    1: [0, 0, 255],      # Blue - Cortex
    2: [0, 255, 0],      # Green - Phloem Fibers
    3: [128, 0, 128],    # Purple - Phloem
    4: [255, 0, 0],      # Red - Xylem vessels
    5: [255, 255, 0],    # Yellow - Air-based Pith cells
    6: [255, 165, 0],    # Orange - Water-based Pith cells
}

CLASS_NAMES = {
    0: "background",
    1: "Cortex",
    2: "Phloem Fibers",
    3: "Phloem",
    4: "Xylem vessels",
    5: "Air-based Pith cells",
    6: "Water-based Pith cells",
}

IGNORE_CLASSES = [255]

def mask_to_color(mask_array):
    """
    Convert a grayscale mask to RGB using class colors.
    Ignored classes (0, 255) are shown in black.
    
    Args:
        mask_array: 2D numpy array with class indices
    
    Returns:
        RGB image as numpy array
    """
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
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
    # Calculate percentiles
    lower = np.percentile(img_array, lower_percentile)
    upper = np.percentile(img_array, upper_percentile)
    
    # Clip and normalize
    img_clipped = np.clip(img_array, lower, upper)
    
    if upper > lower:
        img_normalized = ((img_clipped - lower) / (upper - lower) * 255).astype(np.uint8)
    else:
        img_normalized = img_array.astype(np.uint8)
    
    return img_normalized

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
        
        # Load image
        with Image.open(img_path) as img:
            # Normalize 16-bit or float to 8-bit BEFORE conversion
            if img.mode not in ['RGB', 'L', 'RGBA']:
                if img.mode in ['I', 'I;16', 'F']:
                    img_array = np.array(img)
                    img_array = ((img_array - img_array.min()) / 
                                (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='L')
                else:
                    img = img.convert('L')
            
            # Convert to RGB
            img = img.convert("RGB")
            img_np = np.array(img)
        
        # Convert to tensor WITHOUT normalization (normalization happens in predict_batch_correct)
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
def predict_batch_correct(model, batch_images):
    """
    Run batch inference matching model.predict() behavior exactly.
    
    Args:
        model: The loaded model
        batch_images: Tensor of shape (B, C, H, W) - raw images [0-1] range
    
    Returns:
        List of predicted masks (one per image) as numpy arrays
    """
    if model.training:
        model.eval()
    
    device = next(model.parameters()).device
    batch_size = batch_images.shape[0]
    pred_masks = []
    
    for i in range(batch_size):
        x = batch_images[i]  # (C, H, W)
        image_h, image_w = x.shape[-2:]
        
        # Convert to float32 if needed (images are already in [0, 1] range from ToTensor)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        
        # Normalize
        x = transforms.functional.normalize(
            x, mean=model.image_normalize["mean"], std=model.image_normalize["std"]
        )
        
        # Resize to crop size (short side)
        crop_size = min(model.image_size)
        x = transforms.functional.resize(x, size=[crop_size])
        x = x.unsqueeze(0)  # (1, C, H', W')
        
        # Forward pass
        logits = model._forward_logits(x)  # (1, K+1, H', W')
        
        # Restrict to known classes
        logits = logits[:, :-1]  # (1, K, H', W')
        
        # Resize back to original size
        logits = F.interpolate(
            logits, size=(image_h, image_w), mode="bilinear"
        )  # (1, K, H, W)
        
        # Get class predictions
        masks = logits.argmax(dim=1)  # (1, H, W)
        
        # Map internal class IDs to actual class IDs
        masks = model.internal_class_to_class[masks]  # (1, H, W)
        
        pred_masks.append(masks[0].cpu().numpy())
    
    return pred_masks

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
    args = parser.parse_args()
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Setup paths
    IMG_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_MASK_DIR = OUTPUT_DIR / "semantic_masks"
    OUTPUT_OVERLAY_DIR = OUTPUT_DIR / "semantic_masks_overlay"
    CKPT_PATH = args.finetuned_checkpoint
    
    # Create output directories (only on rank 0)
    if local_rank == 0:
        OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
        if args.save_overlay:
            OUTPUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Wait for rank 0 to create directories
    dist.barrier()
    
    # Load model
    if local_rank == 0:
        print("Loading model...")
    model = lightly_train.load_model(CKPT_PATH)
    model.eval()
    model = model.to(device)
    if local_rank == 0:
        print(f"Model loaded on {world_size} GPU(s)")
    
    # Setup dataset and dataloader with distributed sampler
    dataset = ImageDataset(IMG_DIR)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if local_rank == 0:
        print(f"Total images: {len(dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Number of GPUs: {world_size}")
        print(f"Images per GPU: {len(dataset) // world_size}")
        print("-" * 70)
    
    # Batch inference + save
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            # Move images to device
            batch_images = batch['image'].to(device)
            
            # Run batch inference (returns list of numpy arrays)
            pred_masks = predict_batch_correct(model, batch_images)
            
            # Process each item in the batch
            for i in range(len(batch['img_name'])):
                img_np = batch['image_np'][i]
                if isinstance(img_np, torch.Tensor):
                    img_np = img_np.cpu().numpy()
                
                pred_mask_np = pred_masks[i]  # Already numpy
                img_name = batch['img_name'][i]
                
                # Get unique classes
                pred_unique = np.unique(pred_mask_np)
                
                if local_rank == 0:
                    print(f"GPU {local_rank}: {img_name}")
                    print(f"  Predicted classes: {pred_unique}")
                
                # Save mask as 1-channel grayscale (class indices)
                mask_save_path = OUTPUT_MASK_DIR / img_name
                Image.fromarray(pred_mask_np.astype(np.uint8), mode='L').save(mask_save_path)
                
                if local_rank == 0:
                    print(f"  Saved mask to: {mask_save_path}")
                
                # Save overlay if requested
                if args.save_overlay:
                    # Convert mask to color for overlay
                    pred_color_mask = mask_to_color(pred_mask_np)
                    
                    # Normalize image for high contrast overlay
                    img_np_normalized = normalize_image_percentile(img_np, lower_percentile=2, upper_percentile=98)
                    
                    # Create overlay with normalized image
                    pred_overlay = (0.6 * img_np_normalized + 0.4 * pred_color_mask).astype(np.uint8)
                    
                    # Save overlay
                    overlay_save_path = OUTPUT_OVERLAY_DIR / img_name
                    Image.fromarray(pred_overlay).save(overlay_save_path)
                    
                    if local_rank == 0:
                        print(f"  Saved overlay to: {overlay_save_path}")
                
                if local_rank == 0:
                    print("-" * 70)
    
    # Wait for all processes to finish
    dist.barrier()
    
    if local_rank == 0:
        print(f"\nInference complete!")
        print(f"Masks saved to: {OUTPUT_MASK_DIR}")
        if args.save_overlay:
            print(f"Overlays saved to: {OUTPUT_OVERLAY_DIR}")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()