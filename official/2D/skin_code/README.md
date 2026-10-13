# [TransNorm: Transformer Provides a Strong Spatial Normalization Mechanism for a Deep Segmentation Model](https://arxiv.org/abs/2207.13415)

The official code for "_TransNorm: Transformer Provides a Strong Spatial Normalization Mechanism for a Deep Segmentation Model_".


> [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate),
[Mohammad Al-Antary](https://scholar.google.co.uk/citations?user=ZWFq_B0AAAAJ&hl=en), [Moein Heidari](https://scholar.google.com/citations?user=mir8D5UAAAAJ&hl=en&oi=sra), and [Dorit Merhof
](https://scholar.google.com/citations?user=JH5HObAAAAAJ&sortby=pubdate), "TransNorm: Transformer Provides a Strong Spatial Normalization Mechanism for a Deep Segmentation Model", download [link](https://arxiv.org/abs/2207.13415).
---
## Updates
- July 17, 2022: Initial release.
- July 14, 2022: Submitted to IEEE Access Journal [Under Review].
---


## Introduction
In this paper, we argue that combining the two descriptors, namely, CNN and Transformer might provide an efficient feature representation, which is at the heart of our research in this paper. Majority of existing CNN-Transformer based networks suffer from a weak construction on
the skip connection section. To this end, we design a two-level attention mechanism based on the Transformer module to adaptively recalibrate the feature
combination on the skip connection path. 

#### Please consider starring us, if you found it useful. Thanks

## Updates
- June 16, 2022: First release (Complete implemenation for [SKin Lesion Segmentation on ISIC 2017](https://challenge.isic-archive.com/landing/2017/), [SKin Lesion Segmentation on ISIC 2018](https://challenge2018.isic-archive.com/) and [SKin Lesion Segmentation on PH2](https://www.fc.up.pt/addi/ph2%20database.html) dataset added.)

This code has been implemented in python language using Pytorch library and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Pytorch


## Run Demo
For training deep model and evaluating on each data set follow the bellow steps:</br>
1- Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `train_skin.py` for training the model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. </br>
4- For performance calculation and producing segmentation result, run `evaluate_skin.py`. It will represent performance measures and will saves related results in `results` folder.</br>

**Notice:**
For training and evaluating on ISIC 2017 and ph2 follow the bellow steps :

**ISIC 2017**- Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18\7`. </br> then Run ` 	Prepare_ISIC2017.py` for data preperation and dividing data to train,validation and test sets. </br>
**ph2**- Download the ph2 dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract it then Run ` 	Prepare_ph2.py` for data preperation and dividing data to train,validation and test sets. </br>
Follow step 3 and 4 for model traing and performance estimation. For ph2 dataset you need to first train the model with ISIC 2017 data set and then fine-tune the trained model using ph2 dataset.



## Quick Overview
![Diagram of the proposed method](https://github.com/rezazad68/transnorm/blob/main/Figures/model-1.png)

### Perceptual visualization of the proposed two-level attention module.
![Diagram of the proposed method](https://github.com/rezazad68/transnorm/blob/main/Figures/attention_model-1.png)


## Results
In bellow, results of the proposed approach illustrated.
</br>
#### SKin Lesion Segmentation


#### Performance Comparision on SKin Lesion Segmentation
In order to compare the proposed method with state of the art appraoches on SKin Lesion Segmentation, we considered Drive dataset.  

Methods (On ISIC 2017) |Dice-Score | Sensivity| Specificaty| Accuracy
------------ | -------------|----|-----------------|---- 
Ronneberger and et. all [U-net](https://arxiv.org/abs/1505.04597)       |0.8159	  |0.8172  |0.9680  |0.9164	  
Oktay et. all [Attention U-net](https://arxiv.org/abs/1804.03999)   |0.8082  |0.7998      |0.9776	  |0.9145
Lei et. all [DAGAN](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300803)   |0.8425	  |0.8363       |0.9716	 |0.9304
Chen et. all [TransU-net](https://arxiv.org/abs/2102.04306)   |0.8123  |0.8263     |0.9577	  |0.9207
Asadi et. all [MCGU-Net](https://arxiv.org/abs/2003.05056)   |0.8927	  |	0.8502      |**0.9855**	  |0.9570	
Valanarasu et. all [MedT](https://arxiv.org/abs/2102.10662)   |0.8037	  |0.8064       |0.9546	  |0.9090
Wu et. all [FAT-Net](https://www.sciencedirect.com/science/article/abs/pii/S1361841521003728)   |0.8500	  |0.8392  |0.9725	  |0.9326
Azad et. all [Proposed TransNorm]()	  |**0.8933** 	| **0.8532**	|0.9859	  |**0.9582**
### For more results on ISIC 2018 and PH2 dataset, please refer to [the paper]()


#### SKin Lesion Segmentation segmentation result on test data

![SKin Lesion Segmentation  result](https://github.com/rezazad68/transnorm/blob/main/Figures/isic2018.png)


### Model weights
You can download the learned weights for each dataset in the following table. 

Dataset |Learned weights (Will be added)
------------ | -------------
[ISIC 2018]() |[TransNorm]()
[ISIC 2017]() |[TransNorm]()
[Ph2]() | [TransNorm]()



### Query
All implementations are done by Reza Azad and Moein Heidari. For any query please contact us for more information.

```python
rezazad68@gmail.com
moeinheidari7829@gmail.com

```

