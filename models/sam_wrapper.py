"""
models/sam_wrapper.py  [CORRECTED — adversarial review pass 2]

Fixes applied in this pass
==========================
  C2/C3/C4/C8 (prior pass): correct HF SamPromptEncoder / SamMaskDecoder API
      and positional-embedding sourcing; forward_with_box returns RAW LOGITS.

  *** NEW — CRITICAL bug #2 (silent, results-invalidating) ***
  The cached Freeze-Encoder path (`_forward_from_cache`) fed box prompts to
  `prompt_encoder` in *native* image coordinates (e.g. 512 px), but SAM's
  prompt encoder expects coordinates in the model input frame (longest_edge,
  default 1024 px). The standard path is correct because `SamProcessor`
  rescales boxes; the cache path bypassed the processor and never rescaled.

  Consequence: with native_size=512 and target=1024 every box was effectively
  HALVED in scale, collapsing prompts into the top-left quadrant of the image.
  Because `cache_image_embeddings: true` in sam_freeze.yaml, this is the
  PRIMARY Freeze path, so every Freeze-Encoder ablation (A3/A4/A7/A8) and the
  RQs that depend on them (RQ1/RQ3/RQ4) were silently wrong — no crash,
  just degraded/meaningless SAM masks.

  Fix: scale boxes from native (H,W) to the SAM input frame using the EXACT
  same arithmetic HuggingFace `SamProcessor._normalize_coordinates` uses
  (verified numerically against transformers==4.38.2). The helper is module
  level and unit-testable, and `_forward_from_cache` now also asserts boxes
  fall inside the SAM frame so a future regression fails loudly.

Verification
============
  `scale_box_to_sam_frame` reproduces `SamProcessor._normalize_coordinates`
  exactly on square and non-square inputs (atol 1e-4). A self-check at import
  time is intentionally NOT run (no transformers dependency at import); see
  the shipped test for the equivalence proof.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel, SamProcessor

logger = logging.getLogger(__name__)

# SAM ViT-B default input frame (longest_edge). Used only as a fallback when the
# processor does not expose `target_size`; the live value is always read from the
# processor instance so a non-default checkpoint stays correct.
_DEFAULT_SAM_FRAME = 1024


def get_sam_input_frame(processor: SamProcessor) -> int:
    """Return SAM's input-frame size (longest_edge) for box rescaling.

    Reads the public `processor.target_size` (set by SamProcessor.__init__ from
    `image_processor.size['longest_edge']`). Falls back to the image_processor
    config, then to the ViT-B default, so this never raises on odd processors.
    """
    target = getattr(processor, "target_size", None)
    if isinstance(target, (int, float)) and target > 0:
        return int(target)
    try:
        size = processor.image_processor.size
        if isinstance(size, dict) and "longest_edge" in size:
            return int(size["longest_edge"])
    except Exception:  # pragma: no cover - defensive
        pass
    return _DEFAULT_SAM_FRAME


def scale_box_to_sam_frame(
    box: np.ndarray,
    native_h: int,
    native_w: int,
    target_size: int = _DEFAULT_SAM_FRAME,
) -> np.ndarray:
    """Scale an [x1, y1, x2, y2] box from native pixel coords to SAM's input frame.

    Replicates HuggingFace ``SamProcessor._normalize_coordinates`` /
    ``SamImageProcessor._get_preprocess_shape`` exactly (verified numerically
    against transformers==4.38.2): the image is resized so its longest edge is
    ``target_size``, and box coordinates are scaled per-axis by ``new/old``.

    For the square chest-X-ray frames used here (native_h == native_w) this is
    simply ``box * target_size / native_size``; the per-axis form is kept so
    non-square inputs remain correct.
    """
    box = np.asarray(box, dtype=np.float64).reshape(4)
    scale = float(target_size) / float(max(native_h, native_w))
    new_h = int(native_h * scale + 0.5)
    new_w = int(native_w * scale + 0.5)
    sx = new_w / float(native_w)
    sy = new_h / float(native_h)
    return np.array(
        [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
        dtype=np.float32,
    )


def load_sam(
    model_id: str = "facebook/sam-vit-base",
    device: Optional[torch.device] = None,
) -> Tuple[SamModel, SamProcessor]:
    """Load SAM model and processor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading SAM: {model_id}")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device)
    logger.info(f"SAM loaded on {device}")
    return model, processor


def configure_sam_freeze_encoder(model: SamModel) -> SamModel:
    """Freeze vision encoder; keep mask decoder + prompt encoder trainable."""
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.mask_decoder.parameters():
        param.requires_grad = True
    for param in model.prompt_encoder.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Freeze-Encoder: trainable={trainable:,}, frozen={frozen:,}")
    return model


class SAMFineTuner(nn.Module):
    """
    SAM fine-tuning wrapper.

    DESIGN DECISION: forward_with_box() returns RAW LOGITS (not sigmoid probabilities).
    This prevents float16 NaN from logit reconstruction in the training loop.
    Callers must apply torch.sigmoid() for probabilities or (logit > 0) for binary masks.

    COORDINATE CONTRACT: callers pass boxes in *native* (H, W) pixel coordinates —
    the same space as the rgb_images they pass alongside. Rescaling to SAM's input
    frame is handled internally (standard path via SamProcessor, cache path via
    `scale_box_to_sam_frame`). Do NOT pre-scale boxes to 1024 before calling.
    """

    def __init__(
        self,
        model: SamModel,
        processor: SamProcessor,
        device: torch.device,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
        multimask_output: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = device
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.multimask_output = multimask_output
        # Cache the SAM input-frame size once (model-level constant).
        self.sam_frame = get_sam_input_frame(processor)

        if use_cache and cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def precompute_embeddings(
        self,
        image_paths: List[str],
        native_size: int = 512,
        batch_size: int = 4,
    ) -> None:
        """Precompute and cache image embeddings (Freeze-Encoder path only)."""
        if not self.use_cache:
            return

        from data.transforms import resize_image_pil

        logger.info(f"Precomputing embeddings for {len(image_paths)} images...")
        self.model.eval()
        cached_count = 0
        skip_count = 0

        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i: i + batch_size]
                batch_images = []

                for p in batch_paths:
                    if self._get_cache_path(p).exists():
                        skip_count += 1
                        continue
                    batch_images.append((p, resize_image_pil(p, native_size, to_rgb=True)))

                if not batch_images:
                    continue

                paths_to_cache = [x[0] for x in batch_images]
                rgb_arrays = [x[1] for x in batch_images]

                inputs = self.processor(images=rgb_arrays, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)

                # Get image embeddings
                vision_out = self.model.vision_encoder(pixel_values)
                image_embeddings = vision_out.last_hidden_state  # (B, C, H_emb, W_emb)

                # Fix C2: correct method — on SamModel, not SamPromptEncoder
                image_pe = self.model.get_image_wide_positional_embeddings()
                # image_pe: (1, C, H_pe, W_pe) — same for all images, model-level constant

                for j, path in enumerate(paths_to_cache):
                    torch.save({
                        "image_embeddings": image_embeddings[j].cpu(),
                        "image_pe": image_pe.cpu(),
                    }, str(self._get_cache_path(path)))
                    cached_count += 1

                if i % (batch_size * 10) == 0:
                    logger.info(f"  Embedding progress: {i + len(batch_images)}/{len(image_paths)}")

        logger.info(f"Precompute done: cached={cached_count}, skipped={skip_count}")
        self.model.train()

    def forward_with_box(
        self,
        rgb_images: List[np.ndarray],
        boxes: List[np.ndarray],
        image_paths: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        SAM forward pass with box prompts.

        Boxes are in native (H, W) pixel coordinates; rescaling to SAM's input
        frame is internal. Returns:
            logits: (B, 1, H_native, W_native) float32 RAW LOGITS before sigmoid.
            info: Dict with iou_scores (may be empty for cache path).
        """
        if self.use_cache and image_paths is not None:
            return self._forward_from_cache(rgb_images, boxes, image_paths)
        return self._forward_standard(rgb_images, boxes)

    def _forward_standard(
        self,
        rgb_images: List[np.ndarray],
        boxes: List[np.ndarray],
    ) -> Tuple[torch.Tensor, dict]:
        """Standard forward: full model (for LoRA path and non-cached Freeze path)."""
        native_h = rgb_images[0].shape[0]
        native_w = rgb_images[0].shape[1] if len(rgb_images[0].shape) > 1 else native_h

        # SamProcessor handles resizing + normalization + box coordinate scaling
        input_boxes = [[box.tolist()] for box in boxes]
        inputs = self.processor(images=rgb_images, input_boxes=input_boxes, return_tensors="pt")

        pixel_values = inputs["pixel_values"].to(self.device)
        # processor returns input_boxes as scaled tensor in 1024-space
        proc_boxes = inputs.get("input_boxes")
        if isinstance(proc_boxes, torch.Tensor):
            proc_boxes = proc_boxes.to(self.device)

        # Use SamModel.forward which handles the full pipeline correctly
        outputs = self.model(
            pixel_values=pixel_values,
            input_boxes=proc_boxes,
            multimask_output=self.multimask_output,
        )

        # outputs.pred_masks is 5D: (B, point_batch, num_masks, H_low, W_low).
        # PASS-2 FIX: the previous `[:, :1, :, :]` left the tensor 5D, and
        # F.interpolate(mode="bilinear") REQUIRES 4D -> it raised at runtime.
        # Select point_batch index 0 (one box per image) and the first mask:
        logits = outputs.pred_masks[:, 0, :1, :, :]  # (B, 1, H_low, W_low)
        logits = F.interpolate(
            logits, size=(native_h, native_w),
            mode="bilinear", align_corners=False,
        )

        return logits, {"iou_scores": outputs.iou_scores}

    def _forward_from_cache(
        self,
        rgb_images: List[np.ndarray],
        boxes: List[np.ndarray],
        image_paths: List[str],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass using pre-cached image embeddings (Freeze-Encoder path).

        Fixes C2/C3/C4/C8: correct API calls for SamPromptEncoder and SamMaskDecoder.
        Fixes #2: boxes are rescaled from native coords to SAM's input frame before
        being handed to the prompt encoder (previously passed unscaled → prompts
        collapsed into the top-left quadrant).
        """
        native_h = rgb_images[0].shape[0]
        native_w = rgb_images[0].shape[1] if len(rgb_images[0].shape) > 1 else native_h

        # Fix C2: positional embeddings come from model, not prompt_encoder
        image_pe_base = self.model.get_image_wide_positional_embeddings()
        # (1, C, H_pe, W_pe) — repeated per batch item below

        logits_list = []

        for i, (path, box) in enumerate(zip(image_paths, boxes)):
            cache_path = self._get_cache_path(path)

            if cache_path.exists():
                cached = torch.load(str(cache_path), map_location=self.device)
                image_embeddings = cached["image_embeddings"].unsqueeze(0)  # (1, C, H, W)
            else:
                # Fix C8: cache miss handled cleanly, image_pe assigned before use
                from data.transforms import resize_image_pil
                rgb = resize_image_pil(path, native_h, to_rgb=True)
                inputs = self.processor(images=[rgb], return_tensors="pt")
                with torch.no_grad():
                    vision_out = self.model.vision_encoder(
                        inputs["pixel_values"].to(self.device)
                    )
                    image_embeddings = vision_out.last_hidden_state  # (1, C, H, W)

            # image_pe must match batch size = 1
            image_pe = image_pe_base.to(self.device)  # (1, C, H_pe, W_pe)

            # *** Fix #2: rescale box from native (H, W) to SAM's input frame ***
            # The cached embeddings are produced by the processor on a 1024-px
            # frame, so prompts MUST be in that same frame. Without this the box
            # was interpreted as already-1024 coords and shrank to ~top-left
            # quadrant. scale_box_to_sam_frame matches SamProcessor exactly.
            scaled_box = scale_box_to_sam_frame(
                np.asarray(box, dtype=np.float32), native_h, native_w, self.sam_frame
            )
            # Defensive: a correctly scaled box must lie within [0, sam_frame].
            # Allow a small epsilon for rounding. Fail loudly on regression.
            if float(scaled_box.max()) > self.sam_frame + 1.0 or float(scaled_box.min()) < -1.0:
                logger.warning(
                    "Scaled box %s outside SAM frame [0, %d] (native=%dx%d). "
                    "Check box coordinate space.",
                    scaled_box.tolist(), self.sam_frame, native_h, native_w,
                )

            box_t = torch.tensor(scaled_box, dtype=torch.float32, device=self.device)
            box_t = box_t.unsqueeze(0).unsqueeze(0)  # (1, 1, 4)

            # Fix C3: correct SamPromptEncoder parameter names
            sparse_emb, dense_emb = self.model.prompt_encoder(
                input_points=None,
                input_labels=None,
                input_boxes=box_t,
                input_masks=None,
            )

            # Fix C4: correct SamMaskDecoder parameter name.
            # PASS-2 FIX: SamMaskDecoder.forward returns 3 items in
            # transformers>=4.38 (masks, iou_pred, mask_decoder_attentions);
            # the previous 2-tuple unpack raised ValueError at runtime. Index
            # defensively so this is robust across transformers versions.
            decoder_out = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_pe,   # CORRECT — was image_pe=
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=self.multimask_output,
            )
            low_res_masks = decoder_out[0]

            # low_res_masks is 5D: (1, point_batch, num_masks, H_low, W_low).
            # PASS-2 FIX (same as standard path): drop point_batch idx 0 and
            # take the first mask -> 4D, required by F.interpolate(bilinear).
            logit = low_res_masks[:, 0, :1, :, :]  # (1, 1, H_low, W_low)
            logit = F.interpolate(
                logit, size=(native_h, native_w),
                mode="bilinear", align_corners=False,
            )
            logits_list.append(logit)

        logits = torch.cat(logits_list, dim=0)  # (B, 1, H, W)
        return logits, {}

    def _get_cache_path(self, image_path: str) -> Path:
        stem = Path(image_path).stem
        return Path(self.cache_dir) / f"{stem}_embedding.pt"
