import logging
import sys
from pathlib import Path

from qlty.qlty2D import NCYXQuilt
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


# NOTE: Deleted crop_image_to_patches in favor of using qlty
class ImageCropper:
    """Crop images into patches using QLTY."""

    def __init__(
        self,
        qlty_window: int,
        qlty_step: int,
        qlty_border: int,
        qlty_border_weight: float,
    ):
        self.qlty_object = None
        self.qlty_window = qlty_window
        self.qlty_step = qlty_step
        self.qlty_border = qlty_border
        self.qlty_border_weight = qlty_border_weight

    def initialize_qlty(self, image_shape: tuple[int, ...]):
        """Initialize QLTY object based on image shape."""
        logger.info(f"Original Image shape: {image_shape}")

        self.qlty_object = NCYXQuilt(
            X=image_shape[-1],
            Y=image_shape[-2],
            window=(self.qlty_window, self.qlty_window),
            step=(self.qlty_step, self.qlty_step),
            border=(self.qlty_border, self.qlty_border),
            border_weight=self.qlty_border_weight,
        )

    def unstitch(self, images: torch.Tensor) -> torch.Tensor:
        """Unstitch images into patches."""
        if self.qlty_object is None:
            self.initialize_qlty(images.shape)

        patched_images = self.qlty_object.unstitch(images)

        logger.info(
            f"Patched Image shape: {patched_images.shape}"
        )

        return patched_images
    
    def stitch(self, patched_images: torch.Tensor) -> torch.Tensor:
        """Stitch patches back into full images."""
        if self.qlty_object is None:
            raise ValueError("QLTY object is not initialized.")

        stitched_images = self.qlty_object.stitch(patched_images)[0] # Ignore weights

        logger.info(
            f"Stitched Image shape: {stitched_images.shape}"
        )

        return stitched_images


# NOTE: Deleting normalize_for_display function in favot of plotting tif_to_rgb_pil
# TODO: Explore using pytorch for normalization to avoid conversion to numpy
def tif_to_rgb_pil(image_path: str, percentile_low=0.5, percentile_high=99.5) -> Image.Image:
    """
    Convert a TIFF image to RGB PIL Image with proper normalization.
    
    Args:
        image_path: Path to the TIFF file
        percentile_low: Lower percentile for contrast stretching
        percentile_high: Upper percentile for contrast stretching
        
    Returns:
        RGB PIL Image
    """
    img = Image.open(image_path)
    
    # Handle different image modes with proper normalization
    if img.mode not in ['RGB', 'L', 'RGBA']:
        if img.mode in ['I', 'I;16', 'F']:
            # For 16-bit or float images, apply percentile-based contrast stretching
            img_array = np.array(img).astype(np.float32)
            
            p_low = np.percentile(img_array, percentile_low)
            p_high = np.percentile(img_array, percentile_high)
            
            img_array = np.clip(img_array, p_low, p_high)
            # TODO: Can we avoid conversion to uint8 here?
            img_array = ((img_array - p_low) / (p_high - p_low) * 255).astype(np.uint8)
            
            img = Image.fromarray(img_array, mode='L')
        else:
            img = img.convert('L')
    
    # TODO: Do we need to convert to RGB at all?
    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    return img


def load_images_from_dir(input_dir: str) -> list[str]:
    """Load image file paths from a directory based on given extensions."""
    input_path = Path(input_dir)
    tiff_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))

    if not tiff_files:
        logger.error(f"Error: No TIFF files found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(tiff_files)} TIFF files in {input_dir}")

    # Load all images
    logger.info("Loading images...")
    image_tensors = []
    
    for tiff_file in tiff_files:
        logger.info(f"  Loading {tiff_file.name}")
        image_pil = tif_to_rgb_pil(str(tiff_file))
        image_arr = np.array(image_pil)
        
        # Convert to tensor (C, H, W) and normalize to [0, 1]
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        image_tensors.append(image_tensor)
    
    # Stack into batch (N, C, H, W)
    images_tensor = torch.stack(image_tensors, dim=0)
    logger.info(f"Loaded {len(tiff_files)} images with shape: {images_tensor.shape}")
    return images_tensor