"""
models/lora_sam.py

Custom LoRA adaptation for SAM's vision encoder.

Why not use PEFT?
  transformers.SamModel does not register a PEFT TaskType and SAM's
  attention uses a fused QKV projection (single nn.Linear for all three).
  PEFT's automatic module discovery fails on this architecture.
  This custom implementation is robust and fully traceable.

Audit fix: CRITICAL-04 — custom LoRA instead of PEFT for SAM compatibility.
Audit fix: CRITICAL-05 — manual gradient checkpointing via torch.utils.checkpoint.

Reference:
  Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  ICLR 2022.
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for a single nn.Linear layer.

    Adds trainable low-rank matrices A and B alongside the frozen original.
    During forward: output = W(x) + (B @ A)(x) * (alpha / rank)

    Args:
        original_linear: The frozen linear layer to adapt.
        rank: LoRA rank. Higher rank → more capacity but more parameters.
        alpha: LoRA scaling factor. Effective scale = alpha / rank.
        dropout: Dropout on the LoRA branch.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices: A projects down, B projects up
        # Initialize A with kaiming_uniform (same as nn.Linear default)
        # Initialize B with zeros (so LoRA starts as identity perturbation)
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Freeze the original linear layer
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: frozen original + LoRA perturbation.

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        original_out = self.original(x)
        # LoRA branch: x → dropout → A → B → scale
        lora_out = (
            self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        ) * self.scaling
        return original_out + lora_out


def inject_lora_into_sam(
    sam_model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_module_name: str = "qkv",
) -> nn.Module:
    """
    Inject LoRA into all matching linear layers in SAM's vision encoder.

    Traverses the vision encoder's transformer blocks and replaces
    all nn.Linear modules whose name matches target_module_name.

    For facebook/sam-vit-base, the fused QKV projection is at:
        model.vision_encoder.layers[i].attn.qkv

    This function also freezes all vision encoder parameters
    except the newly injected LoRA parameters.

    Args:
        sam_model: Loaded SamModel from HuggingFace transformers.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        target_module_name: Name of the linear submodule to replace.

    Returns:
        Modified sam_model with LoRA injected.
    """
    # First, freeze entire vision encoder
    for param in sam_model.vision_encoder.parameters():
        param.requires_grad = False

    lora_count = 0

    # Recursively find and replace target modules
    def _replace_recursive(module: nn.Module, parent_name: str = "") -> None:
        nonlocal lora_count
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if (
                name == target_module_name
                and isinstance(child, nn.Linear)
            ):
                # Replace with LoRA wrapper
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer)
                lora_count += 1
                logger.debug(f"LoRA injected at: {full_name}")
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(sam_model.vision_encoder)

    # Also train prompt encoder and mask decoder
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = True
    for param in sam_model.mask_decoder.parameters():
        param.requires_grad = True

    trainable_params = sum(
        p.numel() for p in sam_model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in sam_model.parameters())

    logger.info(
        f"LoRA injection complete: {lora_count} layers replaced | "
        f"trainable={trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return sam_model


def enable_gradient_checkpointing_sam(sam_model: nn.Module) -> None:
    """
    Enable gradient checkpointing for SAM's vision encoder layers.

    This patches each SamVisionEncoderLayer to use
    torch.utils.checkpoint.checkpoint, which recomputes activations
    during the backward pass instead of storing them.

    Effect: ~50-70% reduction in activation memory, ~33% slower training.
    No effect on model weights or final accuracy.

    Audit fix: CRITICAL-05 — manual gradient checkpointing for SAM.

    Args:
        sam_model: SamModel with LoRA already injected.
    """

    def make_checkpoint_forward(original_forward):
        def checkpoint_forward(*args, **kwargs):
            def custom_forward(*inputs):
                return original_forward(*inputs, **kwargs)
            # Use gradient checkpointing (recompute activations on backward)
            return torch.utils.checkpoint.checkpoint(
                custom_forward, *args, use_reentrant=False
            )
        return checkpoint_forward

    patched_count = 0
    for module in sam_model.vision_encoder.modules():
        # Target the per-layer forward pass in SAM's vision encoder
        class_name = type(module).__name__
        if "EncoderLayer" in class_name or "Block" in class_name:
            module.forward = make_checkpoint_forward(module.forward)
            patched_count += 1

    logger.info(
        f"Gradient checkpointing enabled for {patched_count} "
        "vision encoder blocks."
    )


def get_lora_state_dict(sam_model: nn.Module) -> dict:
    """
    Extract only LoRA parameters and trainable decoder/prompt parameters.

    Used for lightweight checkpoint saving (don't save frozen weights).

    Args:
        sam_model: SAM model with LoRA.

    Returns:
        State dict containing only trainable parameters.
    """
    return {
        k: v for k, v in sam_model.state_dict().items()
        if any(
            part in k
            for part in ["lora_A", "lora_B", "mask_decoder", "prompt_encoder"]
        )
    }


def count_trainable_params(model: nn.Module) -> dict:
    """
    Count trainable vs frozen parameters.

    Returns:
        Dict with 'trainable', 'frozen', 'total', 'trainable_pct'.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "trainable_pct": 100 * trainable / total if total > 0 else 0.0,
    }
