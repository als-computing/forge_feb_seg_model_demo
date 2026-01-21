import logging

import torch
import numpy as np

from sam3 import build_sam3_image_model

logger = logging.getLogger(__name__)


def load_state_dict_flexible(ckpt_path: str) -> dict:
    """
    Load a checkpoint and return a clean state_dict (no module. prefix).
    
    Args:
        ckpt_path: Path to checkpoint file
        
    Returns:
        State dict with cleaned keys
        
    Raises:
        ValueError: If checkpoint format is unexpected
    """
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


def to_numpy(x):
    """
    Convert tensor to numpy array safely.
    
    Args:
        x: Tensor or array-like object
        
    Returns:
        Numpy array or None
    """
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return x


def extract_masks_scores(inference_state: dict):
    """
    Extract masks and scores from inference state as tensors.
    
    Args:
        inference_state: Dictionary containing inference results
        
    Returns:
        Tuple of (masks, scores) as torch tensors
    """
    masks = inference_state.get("masks", inference_state.get("pred_masks", None))
    scores = inference_state.get("scores", inference_state.get("pred_logits", None))

    # Ensure they're tensors
    if masks is not None and not isinstance(masks, torch.Tensor):
        masks = torch.from_numpy(np.array(masks))
    
    if scores is not None and not isinstance(scores, torch.Tensor):
        scores = torch.from_numpy(np.array(scores))
    
    # Normalize scores if needed
    if scores is not None and scores.numel() > 0:
        if scores.ndim > 1:
            scores = scores.max(dim=-1)[0]
        # Apply sigmoid if values are logits (outside -1 to 1 range)
        if scores.abs().max() > 10:
            scores = torch.sigmoid(scores)
    
    return masks, scores


def load_model(checkpoint_path, bpe_path, device, original_checkpoint_path=None, is_finetuned=False):
    """
    Load a SAM3 model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        bpe_path: Path to BPE vocabulary file
        device: Device to load model on
        original_checkpoint_path: Path to original checkpoint (needed for finetuned models)
        is_finetuned: Whether this is a finetuned model
        
    Returns:
        Loaded SAM3 model
    """
    state_dict = load_state_dict_flexible(checkpoint_path)
    
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        enable_segmentation=True
    )
    
    if is_finetuned:
        if original_checkpoint_path is None:
            raise ValueError("original_checkpoint_path required for finetuned models")
        # Load original weights first
        original_state = load_state_dict_flexible(original_checkpoint_path)
        _ = model.load_state_dict(original_state, strict=False)
        
        # Then load finetuned weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"⚠ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"⚠ Unexpected keys: {len(unexpected_keys)}")
    else:
        _ = model.load_state_dict(state_dict, strict=False)
    
    model.eval().to(device)
    return model