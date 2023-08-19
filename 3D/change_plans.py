import pickle

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def save_my_plans(file, plans):
        with open(file, 'wb') as f:
            pickle.dump(plans, f)


if __name__ == '__main__':
    plans_file = '/home/leon/repos/deformableLKA/3D/DATASET/d_lka_former_raw/d_lka_former_raw_data/Task02_Synapse/Task002_Synapse/d_lka_former_Plansv2.1_plans_3D.pkl'

    plans = load_pickle(plans_file)

    print("Test")

    plans['data_identifier'] = 'd_lka_former_Data_plans_v2.1'

    save_my_plans('/home/leon/repos/deformableLKA/3D/DATASET/d_lka_former_raw/d_lka_former_raw_data/Task02_Synapse/Task002_Synapse/d_lka_former_Plansv2.1_plans_3D.pkl', plans=plans)

