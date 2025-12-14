from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class BackboneOutput:
    """
    Standardized output for DINOv3 ViT backbones.

    cls: (B, C)
    patch_tokens: (B, N, C)  where N = H_p * W_p
    patch_hw: (H_p, W_p)

    layers (optional):
      dict[layer_idx] -> {"cls": (B,C), "patch_tokens": (B,N,C)}
      where layer_idx refers to hidden_states index:
        0 = embeddings output, 1..L = transformer block outputs
    """
    cls: Tensor
    patch_tokens: Tensor
    patch_hw: Tuple[int, int]
    layers: Optional[Dict[int, Dict[str, Tensor]]] = None


class DinoV3BackboneHF(nn.Module):
    """
    DINOv3 backbone wrapper for Hugging Face Transformers.

    Uses:
      - AutoModel.from_pretrained("facebook/dinov3-...")

    Output conventions (Transformers DINOv3 ViT):
      outputs.last_hidden_state: (B, 1 + R + N, C) where
        token 0 is CLS, next R tokens are register tokens,
        remaining are patch tokens.
      outputs.pooler_output: (B, C) (not used here; we use CLS directly)

    References:
      HF docs show CLS/register/patch splitting and shapes.
    """

    def __init__(
        self,
        model: nn.Module,
        patch_size: int,
        num_register_tokens: int,
        embed_dim: int,
        *,
        normalize_tokens: bool = True,
        use_amp: bool = True,
        return_hidden_states: bool = False,
        return_layers: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.patch_size = int(patch_size)
        self.num_register_tokens = int(num_register_tokens)
        self.embed_dim = int(embed_dim)

        self.normalize_tokens = bool(normalize_tokens)
        self.use_amp = bool(use_amp)

        self.return_hidden_states = bool(return_hidden_states)
        self.return_layers = list(return_layers) if return_layers is not None else []

    @torch.no_grad()
    def forward(self, pixel_values: Tensor) -> BackboneOutput:
        """
        Args:
          pixel_values: (B,3,H,W) already normalized/resized by AutoImageProcessor or your own transforms.

        Returns:
          BackboneOutput (CLS + patch tokens, register tokens removed)
        """
        image_hw = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
        patch_hw = (image_hw[0] // self.patch_size, image_hw[1] // self.patch_size)

        autocast_ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,  # bf16 usually safest with Transformers
            enabled=(self.use_amp and pixel_values.is_cuda),
        )

        with autocast_ctx:
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=self.return_hidden_states or bool(self.return_layers),
                return_dict=True,
            )

        # last_hidden_state: (B, 1 + R + N, C)
        tokens = outputs.last_hidden_state
        cls, patch_tokens = self._split_cls_register_patch(tokens)

        if self.normalize_tokens:
            cls = F.normalize(cls, dim=-1)
            patch_tokens = F.normalize(patch_tokens, dim=-1)

        layers_out = None
        if (self.return_hidden_states or self.return_layers) and getattr(outputs, "hidden_states", None) is not None:
            hs = outputs.hidden_states  # tuple of (B, seq, C)
            wanted = self.return_layers if self.return_layers else range(len(hs))
            layers_out = {}
            for li in wanted:
                if li < 0 or li >= len(hs):
                    continue
                cls_i, patch_i = self._split_cls_register_patch(hs[li])
                if self.normalize_tokens:
                    cls_i = F.normalize(cls_i, dim=-1)
                    patch_i = F.normalize(patch_i, dim=-1)
                layers_out[int(li)] = {"cls": cls_i, "patch_tokens": patch_i}

        return BackboneOutput(cls=cls, patch_tokens=patch_tokens, patch_hw=patch_hw, layers=layers_out)

    def tokens_to_grid(self, patch_tokens: Tensor, patch_hw: Tuple[int, int]) -> Tensor:
        """
        (B, N, C) -> (B, C, H_p, W_p)
        """
        hp, wp = patch_hw
        b, n, c = patch_tokens.shape
        if n != hp * wp:
            raise ValueError(f"N={n} does not match H*W={hp*wp} for patch_hw={patch_hw}")
        return patch_tokens.view(b, hp, wp, c).permute(0, 3, 1, 2).contiguous()

    def upsample_grid_to_image(self, grid: Tensor, image_hw: Tuple[int, int], mode: str = "bilinear") -> Tensor:
        """
        (B, C, H_p, W_p) -> (B, C, H, W)
        """
        return F.interpolate(grid, size=image_hw, mode=mode, align_corners=False if mode in ("bilinear", "bicubic") else None)

    def _split_cls_register_patch(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        """
        tokens: (B, 1 + R + N, C)
        returns:
          cls: (B, C)
          patch_tokens: (B, N, C)
        """
        if tokens.dim() != 3:
            raise ValueError(f"Expected (B,seq,C) tokens, got {tuple(tokens.shape)}")

        cls = tokens[:, 0, :]
        start = 1 + self.num_register_tokens
        patch_tokens = tokens[:, start:, :]
        return cls, patch_tokens


def build_dinov3_hf(
    model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    *,
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    freeze: bool = True,
    normalize_tokens: bool = True,
    use_amp: bool = True,
    return_hidden_states: bool = False,
    return_layers: Optional[Sequence[int]] = None,
) -> DinoV3BackboneHF:
    """
    Builds the backbone from Hugging Face.

    Typical model_ids (see the official collection):
      - facebook/dinov3-vits16-pretrain-lvd1689m
      - facebook/dinov3-vitb16-pretrain-lvd1689m
      - facebook/dinov3-vitl16-pretrain-lvd1689m

    Note: you may need to accept the model license on HF before download.
    """
    from transformers import AutoModel  # local import to keep core light

    model = AutoModel.from_pretrained(model_id, dtype=torch_dtype)

    # DINOv3 ViT config fields (patch_size / num_register_tokens / hidden_size) are documented by HF.
    patch_size = int(getattr(model.config, "patch_size"))
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    embed_dim = int(getattr(model.config, "hidden_size"))

    if device is not None:
        model = model.to(device)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    backbone = DinoV3BackboneHF(
        model=model,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
        embed_dim=embed_dim,
        normalize_tokens=normalize_tokens,
        use_amp=use_amp,
        return_hidden_states=return_hidden_states,
        return_layers=return_layers,
    )
    backbone.eval()
    return backbone
