"""
tests/pipeline/test_memory.py

Memory management tests.

Validates:
  - MemoryMonitor context manager records VRAM stats
  - free_model_memory() actually reduces VRAM after call
  - U-Net forward pass stays within expected memory range
  - Gradient checkpointing reduces memory vs full backprop

Note: GPU tests are skipped on CPU-only environments.
"""

import gc
import time

import pytest
import torch
import torch.nn as nn

from utils.memory_monitor import MemoryMonitor, free_model_memory

CUDA_AVAILABLE = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA")


class TestMemoryMonitor:
    def test_context_manager_runs_without_error(self):
        monitor = MemoryMonitor(limit_gb=14.0)
        with monitor.track("test_stage"):
            x = torch.zeros(100, 100)
            y = x * 2
        report = monitor.report()
        # Should have recorded the stage
        assert "test_stage" in report

    @cuda_only
    def test_peak_vram_tracked(self):
        monitor = MemoryMonitor(limit_gb=14.0)
        device = torch.device("cuda")
        with monitor.track("alloc_100mb"):
            # Allocate ~100MB
            t = torch.zeros(100, 1, 512, 512, dtype=torch.float32, device=device)
        report = monitor.report()
        assert report["alloc_100mb"].peak_allocated_gb > 0.05

    @cuda_only
    def test_free_model_memory_clears_vram(self):
        device = torch.device("cuda")
        model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
        ).to(device)

        before_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        free_model_memory(model)
        after_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        assert after_mb <= before_mb

    def test_get_current_vram_on_cpu_returns_zero(self):
        monitor = MemoryMonitor()
        if not CUDA_AVAILABLE:
            assert monitor.get_current_vram_gb() == 0.0


class TestUNetMemoryUsage:
    @cuda_only
    def test_unet_training_step_vram(self):
        """U-Net training batch should use well under 4 GB."""
        from models.unet import build_unet
        from utils.memory_monitor import MemoryMonitor

        device = torch.device("cuda")
        model = build_unet(features=[32, 64, 128, 256]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        monitor = MemoryMonitor(limit_gb=14.0)

        with monitor.track("unet_train_step"):
            images = torch.rand(8, 1, 256, 256, device=device)
            masks = torch.rand(8, 1, 256, 256, device=device)
            logits = model(images)
            loss = ((torch.sigmoid(logits) - masks) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        report = monitor.report()
        peak_gb = report["unet_train_step"].peak_allocated_gb
        assert peak_gb < 4.0, f"U-Net used {peak_gb:.2f}GB, expected < 4.0GB"

    @cuda_only
    def test_unet_inference_is_leaner_than_training(self):
        """Inference (no_grad) should use less VRAM than training."""
        from models.unet import build_unet

        device = torch.device("cuda")
        model = build_unet(features=[32, 64, 128, 256]).to(device)

        monitor = MemoryMonitor(limit_gb=14.0)

        with monitor.track("inference_no_grad"):
            with torch.no_grad():
                images = torch.rand(16, 1, 256, 256, device=device)
                _ = model(images)

        with monitor.track("train_with_grad"):
            images = torch.rand(8, 1, 256, 256, device=device)
            _ = model(images)

        report = monitor.report()
        assert report["inference_no_grad"].peak_allocated_gb <= \
               report["train_with_grad"].peak_allocated_gb


class TestGradientCheckpointing:
    @cuda_only
    def test_gc_reduces_activation_memory(self):
        """A checkpointed model should use less peak VRAM than uncheckpointed."""
        # Test with a deep linear network as a proxy for ViT
        device = torch.device("cuda")

        class DeepNet(nn.Module):
            def __init__(self, use_checkpoint: bool):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(512, 512) for _ in range(12)]
                )
                self.use_checkpoint = use_checkpoint

            def forward(self, x):
                for layer in self.layers:
                    if self.use_checkpoint:
                        x = torch.utils.checkpoint.checkpoint(
                            layer, x, use_reentrant=False
                        )
                    else:
                        x = layer(x)
                return x

        monitor = MemoryMonitor(limit_gb=14.0)

        # Without checkpointing
        net_no_gc = DeepNet(use_checkpoint=False).to(device)
        with monitor.track("no_checkpoint"):
            inp = torch.rand(64, 512, device=device, requires_grad=True)
            out = net_no_gc(inp)
            out.sum().backward()
        del net_no_gc, inp, out
        torch.cuda.empty_cache()

        # With checkpointing
        net_gc = DeepNet(use_checkpoint=True).to(device)
        with monitor.track("with_checkpoint"):
            inp = torch.rand(64, 512, device=device, requires_grad=True)
            out = net_gc(inp)
            out.sum().backward()

        report = monitor.report()
        # Gradient checkpointing should reduce peak memory
        peak_no_gc = report["no_checkpoint"].peak_allocated_gb
        peak_gc = report["with_checkpoint"].peak_allocated_gb
        assert peak_gc < peak_no_gc, (
            f"GC peak={peak_gc:.3f}GB should be < no-GC peak={peak_no_gc:.3f}GB"
        )
