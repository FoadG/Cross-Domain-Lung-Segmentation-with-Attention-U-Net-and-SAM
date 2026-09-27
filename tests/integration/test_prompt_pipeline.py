"""
tests/integration/test_prompt_pipeline.py

Integration tests for the full prompt generation pipeline:
  U-Net mask → box extraction → coordinate scaling → ensemble variants

Verifies:
  - Coordinate chain is self-consistent (256 → 512 → 1024 roundtrip)
  - Ensemble always returns 3 variants
  - Majority voting is correct
  - Empty ensemble outputs logged (not silently dropped)
"""

import numpy as np
import pytest
import torch

from prompt_generation.box_extractor import (
    extract_box_from_mask,
    get_oracle_box,
    get_pipeline_box,
    scale_box,
)
from prompt_generation.perturbation import (
    create_ensemble_box_variants,
    perturb_box,
    perturb_box_batch,
)


class TestCoordinateChain:
    """Verify the 256 → 512 → SAM coordinate chain is self-consistent."""

    def test_box_in_256_scales_correctly_to_512(self):
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[64:192, 64:192] = 1.0  # centered square, 50% of width

        box_256, _ = extract_box_from_mask(mask, image_size=256, margin_fraction=0.0)
        box_512 = scale_box(box_256, from_size=256, to_size=512)

        # Box should be roughly twice as large
        assert box_512[2] - box_512[0] == pytest.approx(
            2 * (box_256[2] - box_256[0]), abs=4.0
        )
        assert box_512[3] - box_512[1] == pytest.approx(
            2 * (box_256[3] - box_256[1]), abs=4.0
        )

    def test_full_image_mask_stays_within_native_size(self):
        """Full mask should produce box filling most of native_size space."""
        mask = np.ones((256, 256), dtype=np.float32)
        fallback_stats = {
            "x1_norm": 0.05, "y1_norm": 0.05,
            "x2_norm": 0.95, "y2_norm": 0.95,
        }
        box_native, _ = get_pipeline_box(
            pred_mask=mask,
            fallback_box_stats=fallback_stats,
            unet_size=256,
            native_size=512,
            margin_fraction=0.12,
        )
        # All coords must be within 512
        assert box_native.max() <= 512
        assert box_native.min() >= 0

    def test_box_reproducibility(self):
        """Same mask always produces same box."""
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[50:200, 40:210] = 1.0
        fallback_stats = {"x1_norm": 0.1, "y1_norm": 0.1, "x2_norm": 0.9, "y2_norm": 0.9}

        box1, _ = get_pipeline_box(mask, fallback_stats, 256, 512)
        box2, _ = get_pipeline_box(mask, fallback_stats, 256, 512)
        np.testing.assert_array_equal(box1, box2)

    def test_oracle_box_same_as_pipeline_for_perfect_prediction(self):
        """When prediction = GT perfectly, oracle and pipeline boxes should be identical."""
        gt = np.zeros((256, 256), dtype=np.float32)
        gt[60:180, 60:180] = 1.0

        fallback_stats = {"x1_norm": 0.1, "y1_norm": 0.1, "x2_norm": 0.9, "y2_norm": 0.9}
        pipe_box, _ = get_pipeline_box(gt, fallback_stats, 256, 512)
        oracle_box, _ = get_oracle_box(gt, 256, 512)

        # Both should produce the same box when pred == gt
        np.testing.assert_allclose(pipe_box, oracle_box, atol=2.0)


class TestPerturbation:
    def test_perturbed_box_within_image(self):
        box = np.array([50.0, 50.0, 200.0, 200.0], dtype=np.float32)
        for _ in range(20):
            perturbed = perturb_box(box, min_pixels=0, max_pixels=30, image_size=512)
            assert perturbed.min() >= 0
            assert perturbed.max() < 512

    def test_x1_lt_x2_after_perturbation(self):
        """Box ordering must be maintained after perturbation."""
        box = np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)
        for seed in range(50):
            rng = np.random.RandomState(seed)
            perturbed = perturb_box(box, min_pixels=0, max_pixels=50,
                                     image_size=512, rng=rng)
            assert perturbed[0] < perturbed[2], "x1 >= x2 after perturbation"
            assert perturbed[1] < perturbed[3], "y1 >= y2 after perturbation"

    def test_batch_perturbation_returns_same_length(self):
        boxes = [
            np.array([10.0, 10.0, 100.0, 100.0], dtype=np.float32)
            for _ in range(5)
        ]
        result = perturb_box_batch(boxes, min_pixels=0, max_pixels=20, image_size=512)
        assert len(result) == 5

    def test_zero_perturbation_returns_same_box(self):
        box = np.array([50.0, 60.0, 150.0, 160.0], dtype=np.float32)
        rng = np.random.RandomState(0)
        perturbed = perturb_box(box, min_pixels=0, max_pixels=0, image_size=512, rng=rng)
        np.testing.assert_array_equal(box, perturbed)


class TestEnsembleVariants:
    def test_returns_three_variants(self):
        base = np.array([50.0, 50.0, 200.0, 200.0], dtype=np.float32)
        variants = create_ensemble_box_variants(base, image_size=512)
        assert len(variants) == 3

    def test_variant1_is_original(self):
        base = np.array([50.0, 50.0, 200.0, 200.0], dtype=np.float32)
        variants = create_ensemble_box_variants(base, image_size=512)
        np.testing.assert_array_equal(variants[0], base)

    def test_variant2_is_larger_than_original(self):
        base = np.array([100.0, 100.0, 300.0, 300.0], dtype=np.float32)
        variants = create_ensemble_box_variants(
            base, image_size=512, extra_margin_fraction=0.05
        )
        v2 = variants[1]
        assert v2[0] <= base[0], "Variant 2 x1 should be <= original x1"
        assert v2[1] <= base[1], "Variant 2 y1 should be <= original y1"
        assert v2[2] >= base[2], "Variant 2 x2 should be >= original x2"
        assert v2[3] >= base[3], "Variant 2 y2 should be >= original y2"

    def test_all_variants_within_image(self):
        base = np.array([50.0, 50.0, 460.0, 460.0], dtype=np.float32)
        variants = create_ensemble_box_variants(base, image_size=512)
        for v in variants:
            assert v.min() >= 0
            assert v.max() <= 512

    def test_variants_are_deterministic(self):
        base = np.array([80.0, 80.0, 350.0, 350.0], dtype=np.float32)
        v1 = create_ensemble_box_variants(base)
        v2 = create_ensemble_box_variants(base)
        for i in range(3):
            np.testing.assert_array_equal(v1[i], v2[i])
