# Beyond Self-Attention: Deformable Large Kernel Attention for Medical Image Segmentation
Reza Azad<sup>1</sup>, Leon Niggemeier<sup>1</sup>, Michael Hüttemann<sup>1</sup>, Amirhossein Kazerouni<sup>2</sup>, Ehsan Khodapanah Aghdam<sup>3</sup>, Yury Velichko<sup>4</sup>, Ulas Bagci<sup>4</sup>, Dorit Merhof<sup>5,6</sup>

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2309.00121)



## Results
### 2D Synapse Dataset
State-of-the art comparison on the abdominal multi-organ Synapse dataset for 2D methods. For all models the model complexity and the performance (DSC, HD95) is shown. The proposed 2D D-LKA Net achieves superior segmentation performance. Abbreviations stand for: Spl: _spleen_, RKid: _right kidney_, LKid: _left kidney_, Gal: _gallbladder_, Liv: _liver_, Sto: _stomach_, Aor: _aorta_, Pan: _pancreas_. Best results are shown in <h style="color:blue;">blue</h>, second best in <h style="color:red;">red</h>.
![Synapse Results 2D](images/Synapse_2D.png)

### 3D Synapse Dataset
State-of-the art comparison on the abdominal multi-organ Synapse dataset for 3D methods. For all models the model complexity and the performance (DSC, HD95) is shown. The proposed 3D D-LKA Net achieves superior segmentation performance. Our models also is considerably small with the lowest number of parameters. Abbreviations stand for: Spl: _spleen_, RKid: _right kidney_, LKid: _left kidney_, Gal: _gallbladder_, Liv: _liver_, Sto: _stomach_, Aor: _aorta_, Pan: _pancreas_. Best results are shown in <h style="color:blue;">blue</h>, second best in <h style="color:red;">red</h>.
![Synapse Results 3D](images/Synapse_3D.png)

### Qualitative Results
While the 2D version achieves great segmentation results in comparison to other 2D models, the main limitation is the lack of inter-slice connections. Here, the 3D models achieves favorable segmentations.
![Qualitative Results](images/Visualization.png)

## 2D Instructions
For detailed instruction for the 2D methods, please refer to the [Readme](2D/README.md) in the 2D folder.

## 3D Instructions
For detailed instruction for the 3D methods, please refer to the [Readme](3D/README.md) in the 3D folder.


## Updates

## Acknowledgement
This repository is built based on [nnFormer](https://github.com/282857341/nnFormer), [UNETR++](https://github.com/Amshaker/unetr_plus_plus), [transnorm](https://github.com/rezazad68/transnorm), [MCF](https://github.com/WYC-321/MCF), [D3D](https://github.com/XinyiYing/D3Dnet/tree/master). We thank the authors for their code repositories.

## Query

All implementations done by Leon Niggemeier. For any query please contact us for more information.
```python
leon.niggemeier@rwth-aachen.de
```
