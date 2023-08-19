# MCF

## Installation
* CentOS Linux release 7.3.1611
* Python 3.6.13
* CUDA 9.2
* PyTorch 1.9.0
* medpy 0.4.0
* tqdm，h5py

## Getting Started
Please change the database path and data partition file in the corresponding code.
### Dataset
[LA data](https://drive.google.com/drive/folders/1_LObmdkxeERWZrAzXDOhOJ0ikNEm0l_l?usp=sharing)  
[Pancreas data](https://drive.google.com/drive/folders/1kQX8z34kF62ZF_1-DqFpIosB4zDThvPz?usp=sharing)
### Training
`pyhon train_MCF`
### Evaluation
`pyhon test_LA_MCF.py`

## Citation
If you find this project useful, please consider citing:

```bibtex
@InProceedings{MCF,
    author    = {Wang, Yongchao and Xiao, Bin and Bi, Xiuli and Li, Weisheng and Gao, Xinbo},
    title     = {MCF: Mutual Correction Framework for Semi-Supervised Medical Image Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15651-15660}
}
```
## Acknowledgement

We build the project based on UA-MT,SASSNet,DTC.
Thanks for their contribution.
