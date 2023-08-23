# 3D D-LKA Net
Instructions for the 3D D-LKA Net.

## Synapse Dataset

1. Download the Synapse dataset from here: TODO add link
2. Adjust the paths in the **run_training_synappse.sh**
3. Run the following lines: 
    ```bash
    cd 3D
    bash run_training_synapse.sh
    ```
4. After the training is finished, run the evaluation:
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