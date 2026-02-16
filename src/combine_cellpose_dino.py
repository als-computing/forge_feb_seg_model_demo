import numpy as np
import json
from PIL import Image
import cv2
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

def create_class_regions(semantic_mask):
    """
    Create a single mask for each class from the semantic segmentation mask.
    
    Args:
        semantic_mask: numpy array where each pixel value represents a class ID
    
    Returns:
        dict mapping class_id to binary mask (region)
    """
    class_regions = {}
    unique_classes = np.unique(semantic_mask)
    
    for class_id in unique_classes:
        region_mask = (semantic_mask == class_id).astype(np.uint8)
        class_regions[int(class_id)] = region_mask
    
    return class_regions


def assign_objects_to_classes(object_mask, class_regions):
    """
    Assign class IDs to objects based on which region they fall into.
    Uses majority voting with threshold - assigns the class that covers most of the object.
    """
    object_to_class = {}
    unique_objects = np.unique(object_mask)
    unique_objects = unique_objects[unique_objects != 0]
    
    MIN_OVERLAP_THRESHOLD = 30  # Require at least 30% overlap
    
    for obj_id in unique_objects:
        obj_pixels = (object_mask == obj_id)
        obj_area = np.sum(obj_pixels)
        
        best_class = None
        best_overlap = 0
        best_overlap_pct = 0
        
        for class_id, region_mask in class_regions.items():
            if class_id == -1:
                continue
                
            overlap = np.sum(obj_pixels & (region_mask > 0))
            overlap_pct = (overlap / obj_area * 100) if obj_area > 0 else 0
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_pct = overlap_pct
                best_class = class_id
        
        if best_overlap_pct >= MIN_OVERLAP_THRESHOLD:
            object_to_class[int(obj_id)] = best_class
        else:
            object_to_class[int(obj_id)] = None
    
    return object_to_class


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


def save_object_overlay(object_mask, object_to_class, tiff_path, output_path):
    """
    Save objects colored by their assigned class on top of the original image.
    """
    with Image.open(tiff_path) as img:
        if img.mode not in ['RGB', 'L', 'RGBA']:
            if img.mode in ['I', 'I;16', 'F']:
                img_array = np.array(img)
                img_array = normalize_image_percentile(img_array)
                img = Image.fromarray(img_array, mode='L')
            else:
                img = img.convert('L')
        
        if img.mode == 'L':
            img_array = np.array(img)
            img_array = normalize_image_percentile(img_array)
            img = Image.fromarray(img_array, mode='L')
        
        img = img.convert("RGB")
        original_img = np.array(img)
    
    original_img = original_img.astype(np.uint8)
    
    class_colors = {
        0: [128, 128, 128],  # Gray - Background
        1: [0, 0, 255],      # Blue - Cortex
        2: [0, 255, 0],      # Green - Phloem Fibers
        3: [128, 0, 128],    # Purple - Phloem
        4: [255, 0, 0],      # Red - Xylem vessels
        5: [255, 255, 0],    # Yellow - Air-based Pith cells
        6: [255, 165, 0],    # Orange - Water-based Pith cells
    }
    
    colored_objects = np.zeros_like(original_img, dtype=np.uint8)
    
    for obj_id, class_id in object_to_class.items():
        if class_id is not None:
            obj_pixels = (object_mask == obj_id)
            color = class_colors.get(class_id, [128, 128, 128])
            colored_objects[obj_pixels] = color
    
    overlay = original_img.copy()
    object_exists = np.any(colored_objects != 0, axis=2)
    
    alpha = 0.5
    overlay[object_exists] = cv2.addWeighted(
        original_img[object_exists], 
        1-alpha, 
        colored_objects[object_exists], 
        alpha, 
        0
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    
    return overlay


def process_single_file(tiff_path, mask_tiff_path, semantic_mask_path, output_dir):
    """
    Process a single set of files to assign class IDs to objects.
    
    Args:
        tiff_path: path to the original TIFF image
        mask_tiff_path: path to the object mask TIFF (with unique values per object)
        semantic_mask_path: path to the semantic mask TIFF file
        output_dir: directory to save the overlay image
    """
    object_mask = np.array(Image.open(mask_tiff_path))
    semantic_mask = np.array(Image.open(semantic_mask_path))
    
    if len(semantic_mask.shape) == 3 and semantic_mask.shape[2] == 3:
        color_to_class = {
            (128, 128, 128): 0,  # Gray - Background
            (0, 0, 255): 1,      # Blue - Cortex
            (0, 255, 0): 2,      # Green - Phloem Fibers
            (128, 0, 128): 3,    # Purple - Phloem
            (255, 0, 0): 4,      # Red - Xylem vessels
            (255, 255, 0): 5,    # Yellow - Air-based Pith cells
            (255, 165, 0): 6,    # Orange - Water-based Pith cells
        }
        
        class_id_mask = np.zeros(semantic_mask.shape[:2], dtype=np.uint8)
        
        for color, class_id in color_to_class.items():
            mask = np.all(np.abs(semantic_mask - np.array(color)) <= 5, axis=2)
            class_id_mask[mask] = class_id
        
        semantic_mask = class_id_mask
    
    class_regions = create_class_regions(semantic_mask)
    object_to_class = assign_objects_to_classes(object_mask, class_regions)
    
    base_name = Path(tiff_path).stem
    output_path = Path(output_dir) / f"{base_name}_overlay.png"
    save_object_overlay(object_mask, object_to_class, tiff_path, output_path)
    
    return object_to_class, class_regions


def process_file_wrapper(args):
    """
    Wrapper function for multiprocessing that unpacks arguments and handles errors.
    """
    image_path, mask_path, semantic_mask_path, output_dir, index, total = args
    
    print(f"Processing image {index}/{total}: {image_path.name}")
    
    try:
        process_single_file(image_path, mask_path, semantic_mask_path, output_dir)
        return (True, str(image_path.name))
    except Exception as e:
        return (False, str(image_path.name))


def batch_process(images_dir, masks_dir, semantic_masks_dir, output_dir, num_workers=None):
    """
    Process all matching files in the specified directories using multiprocessing.
    
    Args:
        images_dir: directory containing image files (.tiff or .tif)
        masks_dir: directory containing mask files (*_mask.tif or *_mask.tiff)
        semantic_masks_dir: directory containing semantic mask files (*.png or *.tiff)
        output_dir: directory to save overlay images
        num_workers: number of parallel workers (default: CPU count - 1)
    """
    images_dir = Path(images_dir)
    image_files = list(images_dir.glob("*.tiff")) + list(images_dir.glob("*.tif"))
    
    tasks = []
    skipped_files = []
    
    for image_path in sorted(image_files):
        base_name = image_path.stem
        
        masks_dir_path = Path(masks_dir)
        mask_path = masks_dir_path / f"{base_name}_mask.tif"
        if not mask_path.exists():
            mask_path = masks_dir_path / f"{base_name}_mask.tiff"
        
        semantic_masks_dir_path = Path(semantic_masks_dir)
        semantic_mask_path = semantic_masks_dir_path / f"{base_name}.png"
        if not semantic_mask_path.exists():
            semantic_mask_path = semantic_masks_dir_path / f"{base_name}.tiff"
        if not semantic_mask_path.exists():
            semantic_mask_path = semantic_masks_dir_path / f"{base_name}.tif"
        
        if not mask_path.exists() or not semantic_mask_path.exists():
            skipped_files.append(base_name)
            continue
        
        tasks.append((image_path, mask_path, semantic_mask_path, output_dir))
    
    if not tasks:
        print("No files to process!")
        return
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Add index and total to each task
    total = len(tasks)
    tasks_with_index = [(img, mask, sem, out, i+1, total) for i, (img, mask, sem, out) in enumerate(tasks)]
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_file_wrapper, tasks_with_index)
    
    # Summary
    processed_count = sum(1 for success, _ in results if success)
    failed_count = sum(1 for success, _ in results if not success)
    
    print(f"\n{'='*80}")
    print(f"Batch processing complete!")
    print(f"Processed: {processed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Skipped: {len(skipped_files)} files")
    print(f"{'='*80}")
    
    if failed_count > 0:
        print("\nFailed files:")
        for success, filename in results:
            if not success:
                print(f"  - {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine instance masks and semantic masks')
    parser.add_argument('--input-dir', required=True, help='Directory containing input images')
    parser.add_argument('--instance-masks-dir', required=True, help='Directory containing instance masks')
    parser.add_argument('--semantic-masks-dir', required=True, help='Directory containing semantic masks')
    parser.add_argument('--output-dir', required=True, help='Directory to save output overlays')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of parallel workers (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    batch_process(
        args.input_dir,
        args.instance_masks_dir,
        args.semantic_masks_dir,
        args.output_dir,
        args.num_workers
    )