# 3D D-LKA Net
Instructions for the 3D D-LKA Net.

## Environment Setup
1. Create a new conda environment with python version 3.8.16:
    ```bash
    conda create -n "d_lka_net_3d" python=3.8.16
    conda activate d_lka_net_3d
    ```
2. Install PyTorch and torchvision
    ```bash
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    ```
3. Install the requirements with:
    ```bash
    pip install -r requirements.txt
    ```
4. Install 3D deformable convolutions.
    ```bash
    cd dcn/
    bash make.sh
    ```

## Model weights
You can download the learned weights of the D-LKA-Net in the following table. 

Task | Learned weights
------------ | ----
Multi organ segmentation | [D-LKA Net](https://drive.google.com/drive/folders/1Q_V1uNYR7EKkO0dxO8HucD4HgkOfupdc?usp=sharing)
Pancreas | [D-LKA Net](https://drive.google.com/drive/folders/1mSbs-p5gwA2dUbNKJ-xQ08Z717XFbqJ_?usp=sharing)

## Synapse Dataset

1. Download the Synapse dataset from here: [Synapse](https://mbzuaiac-my.sharepoint.com/personal/abdelrahman_youssief_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fabdelrahman%5Fyoussief%5Fmbzuai%5Fac%5Fae%2FDocuments%2FUNETR%2B%2B%2FDATASET%5FSynapse%2Ezip&parent=%2Fpersonal%2Fabdelrahman%5Fyoussief%5Fmbzuai%5Fac%5Fae%2FDocuments%2FUNETR%2B%2B&ga=1)
2. Rename each folder containing 'unetr_pp' to 'd_lka_former'. THIS IS IMPORTANT.
3. Adjust the paths in the **run_training_synappse.sh**
4. Run the following lines: 
    ```bash
    cd 3D
    bash run_training_synapse.sh
    ```
5. After the training is finished, run the evaluation:
    ```bash
    run_evaluation_synapse.sh
    ```
For further instructions, refer to the [nnFormer](https://github.com/282857341/nnFormer) repository.

## Pancreas Dataset
1. Download the pancreas dataset from here: [dataset](https://drive.google.com/drive/folders/1kQX8z34kF62ZF_1-DqFpIosB4zDThvPz)
2. The folder structure should be as follows: 
    ```bash
    /pancreas_code
    --/dataset_pancreas
    ----/Pancreas
    ----/PANCREAS_0001.h5
    .
    .
    .
    ----/PANCREAS_82.h5
    ```
3. Adjust the paths in the **train_pancreas.py** file.
4. Run
    ```bash
    cd 3D/pancreas_code
    python train_pancreas.py
    ```
5. Test
    ```bash
    python test_pancreas.py
    ```
