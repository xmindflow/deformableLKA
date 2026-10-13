"""CPU smoke tests for the upstream 2D D-LKA attention implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "2D"))

from deformable_LKA.deformable_LKA import DeformConv, deformable_LKA_Attention  # noqa: E402
from networks.MaxViT_deform_LKA import MaxViT_deformableLKAFormer  # noqa: E402


def test_deform_conv_zero_offsets_matches_grouped_convolution() -> None:
    torch.manual_seed(7)
    layer = DeformConv(in_channels=4, groups=4, kernel_size=(3, 3), padding=1)
    torch.nn.init.zeros_(layer.offset_net.weight)
    torch.nn.init.zeros_(layer.offset_net.bias)
    inputs = torch.randn(2, 4, 17, 19)

    actual = layer(inputs)
    expected = F.conv2d(inputs, layer.deform_conv.weight, padding=1, groups=4)

    assert actual.shape == (2, 4, 17, 19)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_deformable_lka_attention_preserves_shape_and_backpropagates() -> None:
    torch.manual_seed(11)
    layer = deformable_LKA_Attention(d_model=4)
    inputs = torch.randn(2, 4, 24, 20, requires_grad=True)

    outputs = layer(inputs)
    outputs.square().mean().backward()

    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert layer.spatial_gating_unit.conv0.offset_net.weight.grad is not None


def test_full_segmentation_model_returns_nine_class_logits(monkeypatch) -> None:
    # Upstream unconditionally loads an ignored ImageNet backbone checkpoint;
    # an empty state dict lets this test exercise the architecture with random
    # initialization and without downloading pretrained weights.
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    model = MaxViT_deformableLKAFormer(num_classes=9).eval()
    inputs = torch.zeros(1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (1, 9, 224, 224)
    assert torch.isfinite(outputs).all()
