"""
Combine SAM3 instance masks with DINO semantic masks into a unified labeled mask.

Usage:
    python -m src.combine_sam_dino_v3 \
        --input-dir "$INPUT_DIR" \
        --instance-masks-dir "$SAM3_RESULTS" \
        --semantic-masks-dir "$DINO_RESULTS/semantic_masks" \
        --output-dir "$COMBINED_OUTPUT/sam_dino" \
        --dilate-px 5 \
        --dino-trust Cortex Phloem Air-based_Pith_cells Water-based_Pith_cells

Author: ALS Photon Science Computing
"""

import argparse
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROMPT_TO_SEMANTIC_CLASS: Dict[str, int] = {
    "Cortex":                   1,
    "Phloem_Fibers":            2,
    "Phloem":                   3,
    "Hydrated_Xylem_vessels":   4,
    "Dehydrated_Xylem_vessels": 4,
    "Air-based_Pith_cells":     5,
    "Water-based_Pith_cells":   6,
}

COMBINED_CLASS_ID: Dict[str, int] = {
    "Cortex":                   1,
    "Phloem_Fibers":            2,
    "Phloem":                   3,
    "Hydrated_Xylem_vessels":   4,
    "Dehydrated_Xylem_vessels": 7,
    "Air-based_Pith_cells":     5,
    "Water-based_Pith_cells":   6,
}

CLASS_COLORS_RGB: Dict[str, Tuple[float, float, float]] = {
    "Cortex":                   (0.0,       0.0,       1.0),
    "Phloem_Fibers":            (0.0,       1.0,       0.0),
    "Phloem":                   (128/255,   0.0,       128/255),
    "Hydrated_Xylem_vessels":   (1.0,       0.0,       0.0),
    "Air-based_Pith_cells":     (1.0,       1.0,       0.0),
    "Water-based_Pith_cells":   (1.0,       165/255,   0.0),
    "Dehydrated_Xylem_vessels": (0.0,       199/255,   190/255),  # blue turquoise
}


# =============================================================================
# Core I/O helpers
# =============================================================================
def load_semantic_mask(semantic_path: Path) -> np.ndarray:
    return np.array(Image.open(semantic_path)).astype(np.uint8)


def load_instance_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.max() > 1:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=bool)
    return binary_dilation(mask > 0, structure=struct)


def apply_semantic_filter(mask: np.ndarray, semantic_mask: np.ndarray, class_id: int) -> np.ndarray:
    filtered = mask.copy()
    filtered[semantic_mask != class_id] = 0
    return filtered


def mask_image(image_path: Path, region: np.ndarray) -> np.ndarray:
    img = np.array(Image.open(image_path))
    result = img.copy()
    result[~(region > 0)] = 0
    return result


def save_tif(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path, compression="tiff_deflate")


def find_mask(d: Optional[Path], stem: str) -> Optional[Path]:
    if d is None:
        return None
    for ext in ("_mask.tif", "_mask.tiff"):
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_semantic_mask(semantic_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".tif", ".tiff"):
        p = semantic_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# =============================================================================
# Overlay helpers
# =============================================================================
def _open_as_rgb(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("I", "I;16", "F"):
        arr = np.array(img).astype(np.float32)
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
        denom = (hi - lo) if (hi - lo) > 1e-6 else 1.0
        arr = np.clip((arr - lo) / denom * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def _create_overlay(
    combined: np.ndarray,
    original_path: Path,
    class_colors: Dict[str, Tuple[float, float, float]],
    combined_class_id: Dict[str, int],
    alpha: float,
) -> np.ndarray:
    h, w = combined.shape
    img_arr = _open_as_rgb(original_path)
    if img_arr.shape[:2] != (h, w):
        img_arr = np.array(Image.fromarray(img_arr).resize((w, h), Image.Resampling.BILINEAR))

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    id_to_name = {v: k for k, v in combined_class_id.items()}

    for cid in np.unique(combined):
        if cid == 0:
            continue
        name = id_to_name.get(int(cid))
        if name is None or name not in class_colors:
            continue
        r, g, b = class_colors[name]
        region = combined == cid
        overlay[region, 0] = r
        overlay[region, 1] = g
        overlay[region, 2] = b
        overlay[region, 3] = alpha

    a = overlay[:, :, 3:4]
    result = img_arr * (1 - a) + overlay[:, :, :3] * 255 * a
    return result.astype(np.uint8)


# =============================================================================
# Per-image processing
# =============================================================================
def process_file(
    image_path: Path,
    semantic_dir: Path,
    instance_root: Path,
    out_combined_dir: Path,
    out_overlay_dir: Path,
    dilate_px: int,
    save_extracted: bool,
    output_dir: Path,
    overlay_alpha: float,
    dino_trust: List[str],
) -> None:
    stem = image_path.stem
    try:
        semantic_path = find_semantic_mask(semantic_dir, stem)
        if semantic_path is None:
            logger.warning(f"[{stem}] Semantic mask not found, skipping")
            return
        sem = load_semantic_mask(semantic_path)

        xylem_class_id = PROMPT_TO_SEMANTIC_CLASS["Dehydrated_Xylem_vessels"]  # 4

        hydrated_path   = find_mask(instance_root / "Hydrated_Xylem_vessels",   stem)
        dehydrated_path = find_mask(instance_root / "Dehydrated_Xylem_vessels", stem)

        hydrated_inst   = load_instance_mask(hydrated_path)   if hydrated_path   else None
        dehydrated_inst = load_instance_mask(dehydrated_path) if dehydrated_path else None

        # --- Compute hydrated / dehydrated masks ---
        dehyd_filtered = None
        if dehydrated_inst is not None:
            dehyd_filtered = apply_semantic_filter(dehydrated_inst, sem, xylem_class_id)

        hyd_filtered = None
        if hydrated_inst is not None:
            hyd = hydrated_inst.copy()
            if dehydrated_inst is not None:
                exclude = dilate_mask(dehydrated_inst, dilate_px)
                hyd[exclude] = 0
            else:
                logger.warning(f"[{stem}] Dehydrated mask missing, skipping subtraction")
            hyd_filtered = apply_semantic_filter(hyd, sem, xylem_class_id)

        # --- Xylem_vessels = union of hydrated + dehydrated ---
        xylem_parts = [m for m in [hyd_filtered, dehyd_filtered] if m is not None]
        xylem_union = None
        if xylem_parts:
            xylem_union = xylem_parts[0] if len(xylem_parts) == 1 else \
                ((xylem_parts[0] > 0) | (xylem_parts[1] > 0)).astype(np.float32)
        else:
            logger.warning(f"[{stem}] No xylem masks found, Xylem_vessels will be empty")

        # --- Save extracted per-class masks + images (optional) ---
        if save_extracted:
            for label, mask in [
                ("Xylem_vessels",            xylem_union),
                ("Dehydrated_Xylem_vessels", dehyd_filtered),
                ("Hydrated_Xylem_vessels",   hyd_filtered),
            ]:
                if mask is None:
                    continue
                out_img  = output_dir / label / "images"
                out_mask = output_dir / label / "masks"
                out_img.mkdir(parents=True, exist_ok=True)
                out_mask.mkdir(parents=True, exist_ok=True)
                save_tif((mask * 255).astype(np.uint8), out_mask / f"{stem}_mask.tif")
                save_tif(mask_image(image_path, mask), out_img / f"{stem}.tif")

        # --- Build combined mask ---
        h, w = sem.shape
        combined = np.zeros((h, w), dtype=np.uint8)

        if dehyd_filtered is not None:
            combined[dehyd_filtered > 0] = COMBINED_CLASS_ID["Dehydrated_Xylem_vessels"]

        if hyd_filtered is not None:
            combined[(hyd_filtered > 0) & (combined == 0)] = COMBINED_CLASS_ID["Hydrated_Xylem_vessels"]

        # Non-xylem classes: use DINO directly if in dino_trust, else load SAM3 filtered by DINO
        non_xylem_classes = [
            ("Cortex",                 1),
            ("Phloem_Fibers",          2),
            ("Phloem",                 3),
            ("Air-based_Pith_cells",   5),
            ("Water-based_Pith_cells", 6),
        ]
        for class_name, sem_id in non_xylem_classes:
            if class_name in dino_trust:
                region = (sem == sem_id)
                combined[region & (combined == 0)] = COMBINED_CLASS_ID[class_name]
            else:
                mask_path = find_mask(instance_root / class_name, stem)
                if mask_path is None:
                    logger.warning(f"[{stem}] SAM3 mask not found for {class_name}, skipping")
                    continue
                inst     = load_instance_mask(mask_path)
                filtered = apply_semantic_filter(inst, sem, sem_id)
                combined[(filtered > 0) & (combined == 0)] = COMBINED_CLASS_ID[class_name]

        # --- Save combined mask + overlay ---
        save_tif(combined, out_combined_dir / f"{stem}_combined_mask.tif")
        overlay = _create_overlay(combined, image_path, CLASS_COLORS_RGB, COMBINED_CLASS_ID, overlay_alpha)
        save_tif(overlay, out_overlay_dir / f"{stem}_overlay.tif")

    except Exception as e:
        logger.error(f"[{stem}] Error: {e}")


def process_file_wrapper(args: Tuple) -> None:
    process_file(*args)


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Combine SAM3 instance masks with DINO semantic masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir",          type=str, required=True)
    parser.add_argument("--semantic-masks-dir", type=str, required=True)
    parser.add_argument("--instance-masks-dir", type=str, required=True)
    parser.add_argument("--output-dir",         type=str, default="./sam_dino")
    parser.add_argument("--dilate-px",          type=int, default=8,
                        help="Pixels to dilate dehydrated mask before subtracting")
    parser.add_argument("--overlay-alpha",      type=float, default=0.5,
                        help="Opacity of the color overlay (0=transparent, 1=opaque)")
    parser.add_argument("--dino-trust",         type=str, nargs="+",
                        default=["Cortex", "Phloem_Fibers", "Phloem", "Air-based_Pith_cells", "Water-based_Pith_cells"],
                        choices=["Cortex", "Phloem_Fibers", "Phloem", "Air-based_Pith_cells", "Water-based_Pith_cells"],
                        help="Classes to read directly from DINO semantic mask (rest loaded from SAM3 filtered by DINO)")
    parser.add_argument("--save-extracted",     action="store_true",
                        help="Also save per-class extracted images and masks")
    parser.add_argument("--num-workers",        type=int, default=None)
    args = parser.parse_args()

    input_dir     = Path(args.input_dir)
    semantic_dir  = Path(args.semantic_masks_dir)
    instance_root = Path(args.instance_masks_dir)
    output_dir    = Path(args.output_dir)
    num_workers   = args.num_workers or max(1, cpu_count() - 1)

    out_combined_dir = output_dir / "combined" / "masks"
    out_overlay_dir  = output_dir / "combined" / "overlays"
    out_combined_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    if not image_files:
        logger.error(f"No TIFF images found in {input_dir}")
        return
    logger.info(f"Found {len(image_files)} images | workers={num_workers}")

    tasks = [
        (
            image_path, semantic_dir, instance_root,
            out_combined_dir, out_overlay_dir,
            args.dilate_px, args.save_extracted,
            output_dir, args.overlay_alpha, args.dino_trust,
        )
        for image_path in image_files
    ]

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_wrapper, tasks), total=len(tasks), desc="Processing"))

    logger.info(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()