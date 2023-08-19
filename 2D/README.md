# DAEFormer

The official code for ["_DAE-Former: Dual Attention-guided Efficient Transformer for Medical Image Segmentation_"](https://arxiv.org/abs/2212.13504).

![Proposed Model](./images/proposed_model.png)

## Updates
- 29 Dec., 2022: Initial release with arXiv.
- 27 Dec., 2022: Submitted to MIDL 2023 [Under Review].


## Citation
```
@article{azad2022daeformer,
  title={DAE-Former: Dual Attention-guided Efficient Transformer for Medical Image Segmentation},
  author={Azad, Reza and Arimond, René and Aghdam, Ehsan Khodapanah and Kazerouni, Amirhosein and Merhof, Dorit},
  journal={arXiv preprint arXiv:2212.13504},
  year={2022}
}
```

## How to use

The script train.py contains all the necessary steps for training the network. A list and dataloader for the Synapse dataset are also included.
To load a network, use the --module argument when running the train script (``--module <directory>.<module_name>.<class_name>``, e.g. ``--module networks.DAEFormer.DAEFormer``)





### Model weights
You can download the learned weights of the DAEFormer in the following table. 

Task | Dataset |Learned weights
------------ | -------------|----
Multi organ segmentation | [Synapse](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi) | [DAE-Former](https://drive.google.com/u/0/uc?id=1JEnicYtcMbU_PD_ujCPMaOH5_cs56EIO&export=download)


### Training and Testing

1) Download the Synapse dataset from [here](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi).

2) Run the following code to install the Requirements.

    `pip install -r requirements.txt`

3) Run the below code to train the DAEFormer on the synapse dataset.
    ```bash
    python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5 --batch_size 20 --eval_interval 20 --max_epochs 400 --module networks.DAEFormer.DAEFormer
    ```
    **--root_path**     [Train data path]

    **--test_path**     [Test data path]

    **--eval_interval** [Evaluation epoch]

    **--module**        [Module name, including path (can also train your own models)]
    
 4) Run the below code to test the DAEFormer on the synapse dataset.
    ```bash
    python test.py --volume_path ./data/Synapse/ --output_dir './model_out'
    ```
    **--volume_path**   [Root dir of the test data]
        
    **--output_dir**    [Directory of your learned weights]
    
## Results
Performance comparision on Synapse Multi-Organ Segmentation dataset.

![synapse_results](https://user-images.githubusercontent.com/61879630/210389600-40692e5e-a425-413f-91e6-8e681f2d1532.png)


### Query
All implementation done by Rene Arimond. For any query please contact us for more information.

```python
rene.arimond@lfb.rwth-aachen.de

```
