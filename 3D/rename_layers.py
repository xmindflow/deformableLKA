from d_lka_former.network_architecture.synapse.d_lka_former_synapse import D_LKA_Former
import torch

state_dict_path = '/home/leon/repos/deformableLKA/3D/output_synapse_test_continuing/d_lka_former/3d_fullres/Task002_Synapse/d_lka_former_trainer_synapse__unetr_pp_Plansv2.1/fold_0/model_final_checkpoint_o.model'
copy_dict_path = '/home/leon/repos/deformableLKA/3D/output_synapse_test_continuing/d_lka_former/3d_fullres/Task002_Synapse/d_lka_former_trainer_synapse__unetr_pp_Plansv2.1/fold_0/model_final_checkpoint_key_changed.model'

model_ckpt = torch.load(state_dict_path)
print("Test")

# Rename unetr_pp to d_lka_former
ckpt_copy = model_ckpt
for key in list(model_ckpt['state_dict'].keys()):
    if "unetr_pp" in key:
        print(f"Current key: {key}")
        new_key = key.replace('unetr_pp', 'd_lka_former')
        print(f"Current key: {new_key}")
        ckpt_copy['state_dict'][new_key] = model_ckpt['state_dict'][key]
        ckpt_copy['state_dict'].pop(key)
        print("changed key")

torch.save(ckpt_copy, copy_dict_path)
print("Saved.")