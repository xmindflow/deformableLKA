# D-LKA Net 2D test report

**Date:** 2026-10-13

**Scope:** upstream 2D deformable large-kernel attention operators on CPU

**Environment:** Python 3.9.25, PyTorch 1.13.1, torchvision 0.14.1, CPU

## Test cases

| ID | Case | Expected result |
| --- | --- | --- |
| T1 | Set learned offsets to zero and compare grouped deformable convolution with ordinary grouped convolution | Same output dimensions and values within `rtol=1e-5`, `atol=1e-5` |
| T2 | Run the D-LKA attention block on a non-square batch and backpropagate a scalar loss | Preserve input shape; outputs, input gradients, and deformable-offset gradients are finite and present |
| T3 | Run the full 2D segmentation network on a 224×224 synthetic RGB image with random initialization | Return finite logits of shape `(1, 9, 224, 224)` |

## Result

Command:

```bash
/Volumes/t7/program/python/environments/dlka2d/bin/python -m pytest -q tests/test_dlka_2d.py
```

Result after moving upstream code under `official/`: **3 passed, 1 warning in
4.11s**. The warning is an upstream
`torch.meshgrid` notice about adding the future `indexing` argument.

The test run confirms the local CPU torchvision deformable convolution works,
the attention implementation supports forward and backward passes, and the
full model produces correctly shaped logits. T3 deliberately skips loading the
ImageNet initialization checkpoint. The suite does not load learned D-LKA
weights, evaluate Synapse, or reproduce reported DSC/HD95 values.
See [`DLKA_2D_DATA_AND_ENV.md`](DLKA_2D_DATA_AND_ENV.md) for the upstream
checkpoint and data links and the reason benchmark evaluation is pending.
