# Synapse Partial Evaluation Log

- Model: official `MaxViT_deform_LKA` 2D model
- Checkpoint: `E:\datasets\D-LKA-Net-2D\models\best_model.pth`
- Dataset: Synapse `test_vol_h5` (12 cases total)
- Device: NVIDIA GeForce RTX 4050 on Windows 11
- Status: interrupted after 9 of 12 cases; these are partial results, not a full test-set score.
- Prediction files: not saved (`--is_savenii` was not enabled).
- Provenance: reconstructed from the run output captured in the Codex session. The Windows log file could not be retrieved because SSH was unavailable.

| Case | Mean Dice | Mean HD95 |
|---|---:|---:|
| case0008 | 0.721954 | 8.058252 |
| case0022 | 0.895847 | 21.230212 |
| case0038 | 0.846447 | 12.114614 |
| case0036 | 0.861070 | 4.220779 |
| case0032 | 0.900901 | 4.177124 |
| case0002 | 0.883891 | 5.060130 |
| case0029 | 0.864849 | 3.350815 |
| case0003 | 0.737315 | 91.541600 |
| case0001 | 0.809153 | 19.823636 |
| **Mean of these 9 cases** | **0.835714** | **18.841907** |

The original run successfully loaded the checkpoint and performed GPU inference. It did not train the model. Complete evaluation requires rerunning all 12 cases after the Windows machine is reachable.
