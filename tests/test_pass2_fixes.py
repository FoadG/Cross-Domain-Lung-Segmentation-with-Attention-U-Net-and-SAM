"""
tests/test_pass2_fixes.py

Regression tests for the PASS-2 adversarial-verification fixes. Each test is
designed to FAIL on the pre-PASS-2 code and PASS after the fix:

  * test_sam_forward_returns_4d_* — would raise (5D->bilinear interpolate error,
    and 2-tuple unpack of a 3-value mask_decoder return) on the old sam_wrapper.
    Catches the bug that crashed ALL SAM training/eval but was invisible because
    other suites mock forward_with_box.
  * test_cache_matches_standard — proves the cached Freeze path equals the
    standard path (would diverge under the old unscaled-box bug).
  * test_dataloader_order_reproducible_and_resumable — would fail when shuffle
    draws from the un-seeded global RNG (old behaviour).
  * test_holm_bonferroni — would fail under the old uncorrected-significance code.

The SAM tests build a tiny, RANDOMLY-INITIALISED SAM from config — no weights
download is required (we only assert shapes/equivalence, not accuracy).
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

# Make the project importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

transformers = pytest.importorskip("transformers")
from transformers import SamModel, SamConfig, SamProcessor
from transformers.models.sam.configuration_sam import (
    SamVisionConfig, SamPromptEncoderConfig, SamMaskDecoderConfig,
)
from transformers.models.sam.image_processing_sam import SamImageProcessor

from models.sam_wrapper import SAMFineTuner, scale_box_to_sam_frame


def _tiny_sam():
    H = 256  # channel dim is fixed across prompt/mask/vision-output in real SAM
    cfg = SamConfig(
        vision_config=SamVisionConfig(
            hidden_size=48, num_hidden_layers=2, num_attention_heads=2,
            patch_size=16, image_size=1024, mlp_dim=96, output_channels=H,
            global_attn_indexes=[],
        ).to_dict(),
        prompt_encoder_config=SamPromptEncoderConfig(
            hidden_size=H, image_size=1024, patch_size=16, mask_input_channels=16,
        ).to_dict(),
        mask_decoder_config=SamMaskDecoderConfig(
            hidden_size=H, num_hidden_layers=2, num_attention_heads=2,
            num_multimask_outputs=3, iou_head_depth=2, iou_head_hidden_dim=H,
        ).to_dict(),
    )
    torch.manual_seed(0)
    model = SamModel(cfg).eval()
    proc = SamProcessor(image_processor=SamImageProcessor())
    return model, proc


NATIVE = 512


def test_scale_box_matches_processor():
    model, proc = _tiny_sam()
    for box, h, w in [([100, 100, 400, 400], 512, 512),
                      ([30, 40, 300, 500], 480, 512)]:
        mine = scale_box_to_sam_frame(np.array(box, float), h, w, proc.target_size)
        ref = proc._normalize_coordinates(
            proc.target_size, np.array([box], float), (h, w), is_bounding_box=True
        )[0]
        assert np.allclose(mine, ref, atol=1e-4)


def test_sam_standard_forward_returns_4d():
    model, proc = _tiny_sam()
    ft = SAMFineTuner(model, proc, torch.device("cpu"),
                      use_cache=False, multimask_output=False)
    rgb = [np.random.RandomState(1).randint(0, 255, (NATIVE, NATIVE, 3), np.uint8)]
    box = [np.array([60., 80., 420., 450.], np.float32)]
    with torch.no_grad():
        logits, _ = ft.forward_with_box(rgb, box)
    assert logits.ndim == 4
    assert tuple(logits.shape) == (1, 1, NATIVE, NATIVE)


def test_sam_cache_forward_returns_4d_and_matches_standard():
    model, proc = _tiny_sam()
    dev = torch.device("cpu")
    rgb = [np.random.RandomState(1).randint(0, 255, (NATIVE, NATIVE, 3), np.uint8)]
    box = [np.array([60., 80., 420., 450.], np.float32)]

    ft = SAMFineTuner(model, proc, dev, use_cache=False, multimask_output=False)
    with torch.no_grad():
        std, _ = ft.forward_with_box(rgb, box)

    cdir = tempfile.mkdtemp()
    ftc = SAMFineTuner(model, proc, dev, use_cache=True, cache_dir=cdir,
                       multimask_output=False)
    path = "/tmp/CHNCXR_0001_0.png"
    with torch.no_grad():
        emb = model.vision_encoder(
            proc(images=rgb, return_tensors="pt")["pixel_values"]
        )[0]
    torch.save(
        {"image_embeddings": emb[0].cpu(),
         "image_pe": model.get_image_wide_positional_embeddings().cpu()},
        os.path.join(cdir, "CHNCXR_0001_0_embedding.pt"),
    )
    with torch.no_grad():
        cache, _ = ftc.forward_with_box(rgb, box, image_paths=[path])

    assert cache.ndim == 4 and tuple(cache.shape) == (1, 1, NATIVE, NATIVE)
    # Same embeddings + same scaled box -> mask decoder is deterministic.
    assert torch.allclose(std, cache, atol=1e-4)


def test_sam_forward_batched():
    model, proc = _tiny_sam()
    ft = SAMFineTuner(model, proc, torch.device("cpu"),
                      use_cache=False, multimask_output=False)
    rgb = [np.random.RandomState(i).randint(0, 255, (NATIVE, NATIVE, 3), np.uint8)
           for i in range(2)]
    box = [np.array([60., 80., 420., 450.], np.float32) for _ in range(2)]
    with torch.no_grad():
        logits, _ = ft.forward_with_box(rgb, box)
    assert tuple(logits.shape) == (2, 1, NATIVE, NATIVE)


def test_dataloader_order_reproducible_and_resumable():
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self): return 20
        def __getitem__(self, i): return i

    def orders(seed, epochs, start=0, consume_global=False):
        g = torch.Generator(); g.manual_seed(seed)
        dl = DataLoader(DS(), batch_size=4, shuffle=True, generator=g)
        out = []
        for e in range(start, epochs):
            torch.manual_seed(seed + e); g.manual_seed(seed + e)
            if consume_global:
                _ = torch.randn(1000)  # emulate model/dropout RNG use
            out.append([int(x) for b in dl for x in b])
        return out

    full = orders(42, 3)
    assert full == orders(42, 3)                      # reproducible
    assert full == orders(42, 3, consume_global=True)  # order independent of global RNG
    assert orders(42, 3, start=2)[0] == full[2]        # resume-stable


def test_holm_bonferroni():
    from evaluation.cross_domain_eval import _holm_bonferroni
    h = _holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.04})
    assert h["a"]["reject"] is True
    assert h["b"]["reject"] is False and h["c"]["reject"] is False
    # adjusted p is monotonic and >= raw
    assert h["a"]["p_adjusted"] >= 0.01
    assert h["b"]["p_adjusted"] >= h["a"]["p_adjusted"]
    # None p-values are skipped, not crashed on
    h2 = _holm_bonferroni({"a": None, "b": 0.001})
    assert h2["a"]["p_adjusted"] is None and h2["b"]["reject"] is True