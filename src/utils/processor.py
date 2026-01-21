import torch
from torchvision.transforms import v2


# Define the patched method
@torch.inference_mode()
def set_tensor_batch(self, tensor):
    """
    Set a batch of images from a PyTorch tensor.
    
    Args:
        tensor: PyTorch tensor of shape (B, C, H, W)
    
    Returns:
        state dict with backbone outputs
    """
    state = {}

    B, C, H, W = tensor.shape

    state["original_heights"] = [H] * B
    state["original_widths"] = [W] * B
    state["original_height"] = H
    state["original_width"] = W

    # Transform each image in the batch
    images = [
        self.transform(v2.functional.to_image(tensor[i]).to(self.device))
        for i in range(B)
    ]
    images = torch.stack(images, dim=0)
    
    state["backbone_out"] = self.model.backbone.forward_image(images)
    inst_interactivity_en = self.model.inst_interactive_predictor is not None
    if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
        sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
        sam2_backbone_out["backbone_fpn"][0] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
        )
        sam2_backbone_out["backbone_fpn"][1] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        )
    return state
