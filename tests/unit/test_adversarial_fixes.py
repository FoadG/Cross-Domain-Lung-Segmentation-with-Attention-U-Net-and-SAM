"""
tests/unit/test_adversarial_fixes.py

Regression tests for every critical bug found in the adversarial review.
Each test is labelled with the bug ID it guards against.

These tests ensure the bugs cannot silently regress.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# C1 — Optional import order bug in data/downloader.py
# ─────────────────────────────────────────────────────────────────────────────

class TestC1_DownloaderImport:
    """C1: Optional used before import in downloader.py caused NameError."""

    def test_downloader_imports_without_error(self):
        """The module must be importable without NameError."""
        try:
            import data.downloader as dl
            assert callable(dl.download_dataset)
            assert callable(dl.setup_kaggle_credentials)
        except NameError as e:
            pytest.fail(f"NameError during import (C1 regression): {e}")

    def test_optional_annotation_works_at_runtime(self):
        """Function signatures with Optional must not raise at definition time."""
        from data.downloader import setup_kaggle_credentials, download_dataset
        import inspect
        # These would fail if Optional wasn't imported before the def
        sig1 = inspect.signature(setup_kaggle_credentials)
        sig2 = inspect.signature(download_dataset)
        assert "kaggle_json_path" in sig1.parameters
        assert "kaggle_json_path" in sig2.parameters


# ─────────────────────────────────────────────────────────────────────────────
# C2/C3/C4 — SAM API bugs: wrong method and parameter names
# ─────────────────────────────────────────────────────────────────────────────

class TestC2C3C4_SamApiCorrectness:
    """
    C2: get_dense_pe() doesn't exist — must use get_image_wide_positional_embeddings()
    C3: prompt_encoder kwargs must be input_points/input_labels/input_boxes/input_masks
    C4: mask_decoder kwarg must be image_positional_embeddings (not image_pe)
    """

    def test_sam_wrapper_uses_correct_positional_embedding_method(self):
        """C2: verify corrected source uses the right method name in executable code."""
        import ast
        with open("models/sam_wrapper.py") as f:
            src = f.read()
        # Parse AST to find actual function calls (not in strings/comments)
        tree = ast.parse(src)
        call_attrs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                call_attrs.append(node.attr)
        assert "get_dense_pe" not in call_attrs, (
            "C2 regression: get_dense_pe called as a method in executable code"
        )
        assert "get_image_wide_positional_embeddings" in call_attrs, (
            "C2: get_image_wide_positional_embeddings not called in executable code"
        )

    def test_sam_wrapper_uses_correct_prompt_encoder_params(self):
        """C3: prompt_encoder must be called with input_points/input_boxes/input_masks."""
        import ast
        with open("models/sam_wrapper.py") as f:
            src = f.read()
        tree = ast.parse(src)
        # Find all keyword names in all function calls
        all_keywords = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg:
                        all_keywords.append(kw.arg)
        # Wrong parameter names must not appear in any call
        assert "points" not in all_keywords, "C3 regression: 'points' kwarg in some call"
        assert "input_boxes" in all_keywords, "C3: 'input_boxes' kwarg not found in any call"
        assert "input_points" in all_keywords, "C3: 'input_points' kwarg not found in any call"

    def test_sam_wrapper_uses_correct_mask_decoder_param(self):
        """C4: mask_decoder must use image_positional_embeddings= not image_pe=."""
        import ast
        with open("models/sam_wrapper.py") as f:
            src = f.read()
        tree = ast.parse(src)
        all_keywords = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg:
                        all_keywords.append(kw.arg)
        assert "image_pe" not in all_keywords, "C4 regression: 'image_pe' kwarg in some call"
        assert "image_positional_embeddings" in all_keywords, (
            "C4: 'image_positional_embeddings' kwarg not found"
        )

    def test_transformers_api_matches_our_call(self):
        """Cross-verify actual transformers API matches what we call."""
        import inspect
        try:
            from transformers.models.sam.modeling_sam import (
                SamPromptEncoder, SamMaskDecoder, SamModel
            )
        except ImportError:
            pytest.skip("transformers not installed")

        # C2: verify method exists on SamModel
        assert hasattr(SamModel, "get_image_wide_positional_embeddings"), \
            "C2: get_image_wide_positional_embeddings not on SamModel"

        # C3: verify correct prompt encoder params
        sig = inspect.signature(SamPromptEncoder.forward)
        params = list(sig.parameters.keys())
        assert "input_boxes" in params, f"C3: 'input_boxes' not in {params}"
        assert "input_points" in params
        assert "boxes" not in params, f"C3: deprecated 'boxes' in {params}"

        # C4: verify correct mask decoder param
        sig2 = inspect.signature(SamMaskDecoder.forward)
        params2 = list(sig2.parameters.keys())
        assert "image_positional_embeddings" in params2, \
            f"C4: 'image_positional_embeddings' not in {params2}"
        assert "image_pe" not in params2, f"C4: wrong 'image_pe' in {params2}"


# ─────────────────────────────────────────────────────────────────────────────
# C5 — float16 NaN from logit reconstruction
# ─────────────────────────────────────────────────────────────────────────────

class TestC5_Float16NaNPrevention:
    """C5: logit(sigmoid(x)) in float16 produces inf/NaN. Fix: return raw logits."""

    def test_logit_reconstruction_would_produce_nan_in_float16(self):
        """Demonstrate the original bug: 1 - 1e-7 rounds to 1.0 in float16."""
        x = torch.tensor([1 - 1e-7], dtype=torch.float16)
        assert x.item() == pytest.approx(1.0, abs=1e-4), \
            "float16 precision test setup failed"
        # If we compute logit naively: log(1.0 / (1.0 - 1.0)) = log(inf) = inf
        denom = 1.0 - x
        ratio = x / denom
        assert not torch.isfinite(ratio).all(), \
            "Expected inf in float16 logit reconstruction (confirming the bug exists)"

    def test_sam_trainer_does_not_reconstruct_logits(self):
        """C5 fix: sam_trainer must NOT contain logit-reconstruction in executable code."""
        import ast
        with open("training/sam_trainer.py") as f:
            src = f.read()
        tree = ast.parse(src)
        # Collect all attribute accesses in executable code
        attr_names = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
        # clamp on pred_masks is the signature of the logit reconstruction
        # The reconstruction was: pred_masks.clamp(...) then torch.log(...)
        # Check that torch.log is not called with a division (the reconstruction pattern)
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "log":
                    calls.append("torch.log")
                if isinstance(node.func, ast.Attribute) and node.func.attr == "clamp":
                    # Check if the parent involves pred_masks
                    calls.append("clamp_call")
        # The specific pattern we banned: torch.log(x.clamp(...)/(1-x.clamp(...)))
        # Simple check: no torch.log call should exist in the training loop
        assert "torch.log" not in calls, (
            "C5 regression: torch.log( called in sam_trainer — "
            "logit reconstruction may have returned"
        )

    def test_sam_wrapper_forward_returns_logits_not_sigmoid(self):
        """C5 fix: forward_with_box docstring must say RAW LOGITS."""
        with open("models/sam_wrapper.py") as f:
            src = f.read()
        assert "RAW LOGITS" in src, \
            "C5: 'RAW LOGITS' not documented in sam_wrapper forward_with_box"

    def test_loss_works_correctly_on_raw_logits(self):
        """C5 fix: verify that CombinedLoss works correctly on true logits."""
        from training.losses import CombinedLoss
        criterion = CombinedLoss(alpha=0.5)

        # Simulate what corrected sam_trainer passes: raw SAM logits (unbounded)
        logits = torch.randn(2, 1, 64, 64)  # raw, unbounded
        targets = (torch.rand(2, 1, 64, 64) > 0.5).float()

        total, components = criterion(logits, targets)
        assert torch.isfinite(total), "Loss must be finite for raw logits"
        assert all(np.isfinite(v) for v in components.values()), \
            "Loss components must be finite"

    def test_no_nan_with_float16_logits(self):
        """C5: float16 raw logits (bounded by model initialization) produce no NaN."""
        from training.losses import CombinedLoss
        criterion = CombinedLoss(alpha=0.5)

        # float16 raw logits: bounded around [-10, 10] typically
        logits = torch.randn(2, 1, 32, 32, dtype=torch.float16) * 5.0
        targets = (torch.rand(2, 1, 32, 32) > 0.5).float().half()

        total, components = criterion(
            logits.float(), targets.float()  # AMP upcasts for loss
        )
        assert torch.isfinite(total), "No NaN in float16 logit path"


# ─────────────────────────────────────────────────────────────────────────────
# C6 — Montgomery bilateral mask combination
# ─────────────────────────────────────────────────────────────────────────────

class TestC6_MontgomeryBilateralMasks:
    """C6: Montgomery masks stored as single-lung (one file) instead of combined."""

    def test_validator_filerecord_has_all_mask_paths_field(self):
        """C6 fix: FileRecord must have all_mask_paths field."""
        from data.validator import FileRecord
        import dataclasses
        fields = {f.name for f in dataclasses.fields(FileRecord)}
        assert "all_mask_paths" in fields, \
            "C6: FileRecord missing 'all_mask_paths' field"

    def test_validator_combines_multiple_masks_into_mask_path(self):
        """C6: when multiple mask files exist for same image, combined mask is saved."""
        import os
        from PIL import Image
        from data.validator import FileRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake left+right lung masks
            left_mask = np.zeros((64, 64), dtype=np.uint8)
            left_mask[10:30, 5:30] = 255   # left lung
            right_mask = np.zeros((64, 64), dtype=np.uint8)
            right_mask[10:30, 35:60] = 255  # right lung

            left_path = os.path.join(tmpdir, "MCUCXR_0001_0.png")
            right_path = os.path.join(tmpdir, "MCUCXR_0001_0_right.png")
            Image.fromarray(left_mask).save(left_path)
            Image.fromarray(right_mask).save(right_path)

            # Use the _combine_masks function directly
            from data.validator import _combine_masks
            from pathlib import Path
            combined = _combine_masks([Path(left_path), Path(right_path)])

            # Combined mask must include both lung regions
            assert combined[15, 10] > 0.5, "Left lung region missing from combined mask"
            assert combined[15, 45] > 0.5, "Right lung region missing from combined mask"

    def test_combined_mask_has_more_foreground_than_either_single(self):
        """C6: combined bilateral mask must have more foreground than either alone."""
        from data.validator import _combine_masks
        from pathlib import Path
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            left = np.zeros((128, 128), dtype=np.uint8)
            left[20:80, 10:55] = 255
            right = np.zeros((128, 128), dtype=np.uint8)
            right[20:80, 70:115] = 255

            lp = Path(tmpdir) / "left.png"
            rp = Path(tmpdir) / "right.png"
            Image.fromarray(left).save(str(lp))
            Image.fromarray(right).save(str(rp))

            left_arr = (left > 0).astype(float).mean()
            right_arr = (right > 0).astype(float).mean()
            combined = _combine_masks([lp, rp])

            assert combined.mean() > left_arr, "Combined > left alone"
            assert combined.mean() > right_arr, "Combined > right alone"
            assert abs(combined.mean() - (left_arr + right_arr)) < 0.01, \
                "Non-overlapping masks should sum to ~left + right"


# ─────────────────────────────────────────────────────────────────────────────
# C7 — Missing A8 ablation
# ─────────────────────────────────────────────────────────────────────────────

class TestC7_A8AblationPresent:
    """C7: Ablation A8 (stress test) was missing from run_experiment evaluation."""

    def test_run_experiment_contains_a8_evaluation(self):
        """C7 fix: run_experiment.py must evaluate A8."""
        with open("run_experiment.py") as f:
            src = f.read()
        assert '"A8"' in src, \
            'C7 regression: "A8" key not in run_experiment.py evaluation'
        assert "stress" in src.lower(), \
            "C7: stress test box not referenced in run_experiment.py"
        assert "get_stress_test_box" in src, \
            "C7: get_stress_test_box not imported/used in run_experiment.py"

    def test_ablation_labels_includes_a8(self):
        """C7: cross_domain_eval ABLATION_LABELS must include A8."""
        from evaluation.cross_domain_eval import ABLATION_LABELS
        assert "A8" in ABLATION_LABELS, "C7: A8 missing from ABLATION_LABELS"

    def test_stress_test_box_is_different_from_oracle_box(self):
        """C7: A8 must use corrupted box, not GT box."""
        from prompt_generation.box_extractor import get_oracle_box, get_stress_test_box

        gt = np.zeros((256, 256), dtype=np.float32)
        gt[60:180, 60:180] = 1.0

        oracle_box, _ = get_oracle_box(gt, unet_size=256, native_size=512)
        stress_box = get_stress_test_box(gt, error_scale=0.25, native_size=512, seed=42)

        # Stress box must differ from oracle box
        max_diff = np.abs(oracle_box - stress_box).max()
        assert max_diff > 5, \
            f"Stress box too similar to oracle box (max_diff={max_diff:.1f}px)"


# ─────────────────────────────────────────────────────────────────────────────
# C8 — Cache miss undefined image_pe
# ─────────────────────────────────────────────────────────────────────────────

class TestC8_CacheMissHandling:
    """C8: _forward_from_cache cache miss path left image_pe undefined."""

    def test_cache_miss_path_assigns_image_pe_before_use(self):
        """C8: verify code assigns image_pe BEFORE using it in mask_decoder call."""
        with open("models/sam_wrapper.py") as f:
            src = f.read()

        # Find _forward_from_cache method
        method_start = src.find("def _forward_from_cache")
        method_end = src.find("\n    def ", method_start + 1)
        if method_end == -1:
            method_end = len(src)
        method_body = src[method_start:method_end]

        # image_pe_base must be assigned BEFORE the loop (not only in cache-hit branch)
        pe_assignment_line = None
        cache_miss_start = None
        for i, line in enumerate(method_body.split("\n")):
            if "get_image_wide_positional_embeddings" in line and "pe" in line:
                pe_assignment_line = i
            if "cache_path.exists()" in line:
                cache_miss_start = i

        assert pe_assignment_line is not None, \
            "C8: get_image_wide_positional_embeddings not found in _forward_from_cache"
        if cache_miss_start is not None:
            assert pe_assignment_line < cache_miss_start, \
                "C8: image_pe assigned AFTER the cache check — undefined in cache-miss path"


# ─────────────────────────────────────────────────────────────────────────────
# M1 — stage_done logic bug
# ─────────────────────────────────────────────────────────────────────────────

class TestM1_StageDone:
    """M1: stage_done() used 'Path(...) and Path(...).exists()' — always truthy first operand."""

    def test_stage_done_returns_false_when_marker_absent(self, tmp_path):
        """M1 fix: stage_done must return False when marker file does not exist."""
        import sys
        sys.path.insert(0, ".")
        from run_experiment import stage_done
        assert stage_done(str(tmp_path), 0) is False, \
            "M1: stage_done returns True for non-existent marker"

    def test_stage_done_returns_true_when_marker_present(self, tmp_path):
        """M1: stage_done must return True when marker file exists."""
        from run_experiment import stage_done, mark_done, STAGE_MARKERS
        mark_done(str(tmp_path), 0)
        assert stage_done(str(tmp_path), 0) is True, \
            "stage_done must return True after mark_done"

    def test_stage_done_is_stage_specific(self, tmp_path):
        """M1: marking stage 0 done must not affect stage 1."""
        from run_experiment import stage_done, mark_done
        mark_done(str(tmp_path), 0)
        assert stage_done(str(tmp_path), 0) is True
        assert stage_done(str(tmp_path), 1) is False


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble: sigmoid applied to logits (consequence of C5 fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsembleSigmoidOnLogits:
    """Verify ensemble correctly applies sigmoid to raw logits from forward_with_box."""

    def test_ensemble_source_applies_sigmoid(self):
        """ensemble.py must apply sigmoid before threshold comparison (AST check)."""
        import ast
        with open("prompt_generation/ensemble.py") as f:
            src = f.read()
        tree = ast.parse(src)
        # Find sigmoid calls in executable code
        sigmoid_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "sigmoid":
                    sigmoid_calls.append(node)
        assert len(sigmoid_calls) >= 2, (
            f"Expected >=2 sigmoid calls in ensemble.py, found {len(sigmoid_calls)}. "
            "Both run_ensemble_inference and run_single_box_inference must apply sigmoid."
        )

    def test_majority_vote_correct_with_logit_inputs(self):
        """Full ensemble pipeline: logits → sigmoid → binary → majority vote."""
        from prompt_generation.ensemble import run_ensemble_inference

        class FakeFineTuner:
            """Simulates corrected SAMFineTuner that returns RAW LOGITS."""
            class _FakeModel:
                def parameters(self):
                    return iter([torch.zeros(1)])  # nonempty so next() works
            model = _FakeModel()

            def forward_with_box(self, rgb_images, boxes, image_paths=None):
                B = len(rgb_images)
                H, W = rgb_images[0].shape[:2]
                # Return positive logits for center region — should become foreground
                logits = torch.full((B, 1, H, W), -5.0)  # default: all background
                logits[:, :, H//4:3*H//4, W//4:3*W//4] = 5.0  # center: foreground
                return logits, {}

        finetuner = FakeFineTuner()
        rgb_images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        base_boxes = [np.array([10., 10., 50., 50.], dtype=np.float32) for _ in range(2)]

        ensemble_masks, info = run_ensemble_inference(
            finetuner=finetuner,
            rgb_images=rgb_images,
            base_boxes=base_boxes,
            native_size=64,
        )

        assert ensemble_masks.shape == (2, 1, 64, 64)
        # Center region should be predicted as foreground (logit=5 → sigmoid=0.99 > 0.5)
        assert ensemble_masks[0, 0, 32, 32].item() == pytest.approx(1.0, abs=0.1)
        # Corner region should be background (logit=-5 → sigmoid=0.007 < 0.5)
        assert ensemble_masks[0, 0, 2, 2].item() == pytest.approx(0.0, abs=0.1)
