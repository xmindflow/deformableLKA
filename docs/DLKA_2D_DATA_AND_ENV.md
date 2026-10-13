# D-LKA Net 2D: files, data, and environment

## Upstream materials

- Paper PDF: [`Beyond_Self-Attention_D-LKA_WACV2024.pdf`](../Beyond_Self-Attention_D-LKA_WACV2024.pdf)
- Official implementation: <https://github.com/xmindflow/deformableLKA>
- Official 2D instructions: <https://github.com/xmindflow/deformableLKA/blob/main/2D/README.md>
- MaxViT backbone checkpoint referenced by the official code: <https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth>
- WACV 2024 landing page: <https://openaccess.thecvf.com/content/WACV2024/html/Azad_Beyond_Self-Attention_Deformable_Large_Kernel_Attention_for_Medical_Image_Segmentation_WACV_2024_paper.html>
- PDF SHA-256: `e5a53dd6c9180e067d8e1b85b0ad264349c6f515d23a828e1b5ab6e00f9a2527`

The upstream repository contains both 2D and 3D implementations. The current
work targets the 2D implementation under `2D/`.

## Environment on this Mac

The official 2D README specifies Python 3.8.16, PyTorch 1.11.0, torchvision
0.12.0, and CUDA 11.3. This machine is Apple Silicon with no CUDA device, so
that GPU training environment is not applicable. A CPU test environment has
been created at:

`/Volumes/t7/program/python/environments/dlka2d`

It uses Python 3.9.25 and reuses the preinstalled CPU PyTorch 1.13.1,
torchvision 0.14.1, and timm 0.6.13 from the existing
`/Volumes/t7/program/python/environments/segformer` environment. Additional
2D/evaluation dependencies are pinned in [`requirements-test.txt`](../requirements-test.txt).
Recreate/update it with:

```bash
bash scripts/create_dlka2d_env.sh
```

Run CPU smoke tests with:

```bash
/Volumes/t7/program/python/environments/dlka2d/bin/python -m pytest -q tests
```

For faithful training or paper-metric reproduction, use a Linux host with the
upstream CUDA versions and compatible GPU. The current CPU checks do not claim
to reproduce benchmark scores.

## Data and checkpoints

The upstream 2D README links its Synapse train/test bundle from Google Drive
and links the 2D Synapse checkpoint folder from Google Drive:

- Synapse bundle: <https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi>
- 2D weights folder: <https://drive.google.com/drive/folders/1TY7G0X32kGbgnzx_Zn5px0gW8fVF4ptI?usp=sharing>

Those Google Drive endpoints did not respond from this machine during setup.
No dataset archive, D-LKA checkpoint, or ImageNet backbone checkpoint is
currently present. Keep downloaded data and weights on the external SSD,
outside Git. The official README expects the
Synapse layout `data/Synapse/train_npz` and `data/Synapse/test_vol_h5`; its
training example uses `--root_path ./data/Synapse/train_npz` and
`--test_path ./data/Synapse/test_vol_h5`.

The official 2D README lists no released weights for skin-lesion datasets.
For ISIC data, use the official challenge page and comply with its dataset
terms: <https://challenge.isic-archive.com/data/>. PH2 is separately licensed;
obtain it from its official source before use. Do not commit medical images,
labels, or downloaded checkpoints to this repository.
