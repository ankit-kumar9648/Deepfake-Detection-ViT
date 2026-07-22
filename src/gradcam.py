"""
src/gradcam.py
===============
Explainable AI (STEP 13) for the ViT classifier:

  * Grad-CAM  -- gradient-weighted activations of the last transformer
                 block, reshaped from the patch-token grid back into a
                 224x224 heatmap.
  * Attention rollout -- multiplies the [CLS]-token attention weights
                 across every encoder layer to show which patches the
                 model attended to.

Both return a PIL.Image overlay that can be displayed directly in
Streamlit or saved to disk.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config


def _grid_size(num_patch_tokens: int) -> int:
    """ViT-B/16 @ 224 -> 14x14 patch grid (196 tokens)."""
    side = int(num_patch_tokens ** 0.5)
    return side


def generate_gradcam(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Compute a Grad-CAM heatmap using gradients of the target class logit
    w.r.t. the last hidden state (patch tokens) of the ViT backbone.
    """
    model.eval()
    pixel_values = pixel_values.to(config.DEVICE).requires_grad_(True)

    activations = {}
    gradients = {}

    def fwd_hook(_module, _inp, output):
        activations["value"] = output

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # Target the last encoder layer inside model.backbone
    if config.MODEL_BACKEND == "torchvision":
        target_layer = model.backbone.encoder.layers[-1]
    else:
        target_layer = model.backbone.encoder.layer[-1]

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        # Call model forward; handle optional attention tuple output
        out = model(pixel_values)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        model.zero_grad(set_to_none=True)
        score = logits[0, target_class]
        score.backward()

        acts = activations["value"]          # (1, tokens, hidden)
        grads = gradients["value"]            # (1, tokens, hidden)

        if isinstance(acts, tuple):
            acts = acts[0]

        # Drop the [CLS] token, keep patch tokens only
        patch_acts = acts[0, 1:, :]           # (num_patches, hidden)
        patch_grads = grads[0, 1:, :]         # (num_patches, hidden)

        weights = patch_grads.mean(dim=0)     # (hidden,) global-avg-pool of gradients
        cam = torch.relu((patch_acts * weights).sum(dim=-1))  # (num_patches,)

        side = _grid_size(cam.shape[0])
        cam = cam[: side * side].reshape(side, side).detach().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def generate_attention_rollout(attentions) -> np.ndarray:
    """
    Attention rollout across all encoder layers (Abnar & Zuidema, 2020).
    """
    with torch.no_grad():
        result = torch.eye(attentions[0].shape[-1]).to(attentions[0].device)
        for attn in attentions:
            attn_heads_avg = attn.mean(dim=1)[0]           # (tokens, tokens)
            attn_heads_avg = attn_heads_avg + torch.eye(attn_heads_avg.shape[-1]).to(attn_heads_avg.device)
            attn_heads_avg = attn_heads_avg / attn_heads_avg.sum(dim=-1, keepdim=True)
            result = attn_heads_avg @ result

        cls_attention = result[0, 1:]                       # attention from CLS to patches
        side = _grid_size(cls_attention.shape[0])
        cls_attention = cls_attention[: side * side].reshape(side, side).cpu().numpy()

        if cls_attention.max() > 0:
            cls_attention = (cls_attention - cls_attention.min()) / (
                cls_attention.max() - cls_attention.min() + 1e-8
            )
        return cls_attention


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Resize a (H, W) heatmap to the original image size and alpha-blend it."""
    import matplotlib.cm as cm

    original_image = original_image.convert("RGB").resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE)
    )
    heatmap_t = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(
        heatmap_t, size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        mode="bilinear", align_corners=False,
    )[0, 0].numpy()

    colored = cm.get_cmap(colormap)(heatmap_resized)[:, :, :3]  # drop alpha channel
    colored = (colored * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored)

    blended = Image.blend(original_image, heatmap_img, alpha=alpha)
    return blended