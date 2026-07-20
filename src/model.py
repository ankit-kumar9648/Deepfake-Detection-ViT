"""
src/model.py
============
Pretrained Vision Transformer (ViT-B/16) with a replaced classification
head, using transfer learning (STEP 7). Two interchangeable backends are
supported via config.MODEL_BACKEND:

  * "huggingface" -> transformers.ViTModel("google/vit-base-patch16-224")
  * "torchvision" -> torchvision.models.vit_b_16(weights=IMAGENET1K_V1)

Both are wrapped in a single nn.Module (`DeepfakeViT`) exposing an
identical forward() -> logits interface, and both expose the last
attention layer needed for Grad-CAM / attention-rollout visualisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class ClassifierHead(nn.Module):
    """Small MLP head replacing the ImageNet classification head."""

    def __init__(self, in_features: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HuggingFaceViTClassifier(nn.Module):
    """Wraps transformers ViTModel with a custom classifier head."""

    def __init__(
        self,
        model_name: str = config.HF_MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        freeze_backbone: bool = config.FREEZE_BACKBONE,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        from transformers import ViTModel  # local import keeps config.py light

        self.backbone = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden_size = self.backbone.config.hidden_size
        self.head = ClassifierHead(hidden_size, num_classes, dropout)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = config.UNFREEZE_LAST_N_BLOCKS) -> None:
        """Unfreeze the last `n` transformer encoder blocks for fine-tuning."""
        blocks = self.backbone.encoder.layer
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        # Also unfreeze the final layernorm
        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor, output_attentions: bool = False):
        outputs = self.backbone(pixel_values=pixel_values, output_attentions=output_attentions)
        cls_token = outputs.last_hidden_state[:, 0, :]   # [CLS] token embedding
        logits = self.head(cls_token)
        if output_attentions:
            return logits, outputs.attentions
        return logits


class TorchvisionViTClassifier(nn.Module):
    """Wraps torchvision's vit_b_16 with a custom classifier head."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        freeze_backbone: bool = config.FREEZE_BACKBONE,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.backbone = vit_b_16(weights=weights)
        hidden_size = self.backbone.hidden_dim
        self.backbone.heads = nn.Identity()   # strip ImageNet head
        self.head = ClassifierHead(hidden_size, num_classes, dropout)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = config.UNFREEZE_LAST_N_BLOCKS) -> None:
        blocks = self.backbone.encoder.layers
        for block in list(blocks)[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.encoder.ln.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor, output_attentions: bool = False):
        features = self.backbone(pixel_values)
        logits = self.head(features)
        if output_attentions:
            return logits, None   # torchvision backend does not expose attentions
        return logits


def build_model() -> nn.Module:
    """Factory that returns the configured backend, ready for transfer learning."""
    if config.MODEL_BACKEND == "torchvision":
        model = TorchvisionViTClassifier()
    else:
        model = HuggingFaceViTClassifier()
    return model.to(config.DEVICE)


def get_param_groups(model: nn.Module):
    """
    Discriminative learning rates: a low LR for any unfrozen backbone
    parameters and a higher LR for the freshly-initialised classifier head.
    """
    backbone_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and "head" not in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "head" in n]
    groups = [{"params": head_params, "lr": config.HEAD_LEARNING_RATE}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": config.LEARNING_RATE})
    return groups
