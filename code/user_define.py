from datetime import datetime
import json
import os
from pytz import timezone
import numpy as np
import random


class experiment_config():
    exp_id = 1
    seed = 42
    label_dict = {'Normal': 0, 'Tumor': 1}
    label_dict_inv = {v: k for k, v in label_dict.items()}  # {0: 'Normal', 1: 'Tumor'}
    classes = list(label_dict.keys())
    # ========================================================================================================
    patch_size = 256
    max_lr = 5e-4
    level = 0
    train_batch_size = 32
    inference_batch_size = 256
    epochs = 500
    num_workers = 8
    pin_memory = True
    # ========================================================================================================
    dataset_path = '/CAMELYON16/data/'
    normal_list = []  # normal slides are not used in AL experiments
    # normal_list = list(range(1, 27)) + list(range(28, 45)) + list(range(46, 86)) + list(range(87, 144)) + list(range(145, 161)) # normal_027/045.tif are not used for training as they contain very little tissue, normal_086.tif doesn't exist, normal_144.tif has black background
    tumor_list = list(range(1, 112))
    test_list = list(range(1, 49)) + list(range(50, 114)) + list(range(115, 131))
    # ========================================================================================================
    r = random.Random(seed)
    r.shuffle(tumor_list)
    r.shuffle(normal_list)
    train_valid_ratio = 0.7
    train_n_tumor = int(len(tumor_list) * train_valid_ratio)
    train_n_normal = int(len(normal_list) * train_valid_ratio)

    # e.g., ["tumor_001.tif", "tumor_002.tif"..., "normal_001.tif", "normal_002.tif"...]
    train_list = ["tumor_" + str(n).zfill(3) + ".tif" for n in tumor_list[:train_n_tumor]] + \
                 ["normal_" + str(n).zfill(3) + ".tif" for n in normal_list[:train_n_normal]]
    print(f"Dataset: {len(train_list)} train slides: \n{np.array(train_list)}")

    valid_list = ["tumor_" + str(n).zfill(3) + ".tif" for n in tumor_list[train_n_tumor:]] + \
                 ["normal_" + str(n).zfill(3) + ".tif" for n in normal_list[train_n_normal:]]
    print(f"Dataset: {len(valid_list)} valid slides: \n{np.array(valid_list)}")
    # ========================================================================================================
    tumor_test_list = ['001', '002', '004', '008', '010', '011', '013', '016',
                       '021', '026', '027', '029', '030', '033', '038', '040',
                       '046', '048', '051', '052', '061', '064', '065', '066',
                       '068', '069', '071', '073', '074', '075', '079', '082',
                       '084', '090', '092', '094', '097', '099', '102', '104',
                       '105', '108', '110', '113', '116', '117', '121', '122']
    # 21 macro tumor
    macro_tumor_test_list = ['001', '002', '016', '021', '026', '027', '030',
                             '040', '051', '061', '068', '071', '073', '075',
                             '082', '090', '094', '104', '105', '113', '121']
    # 27 micro tumor
    micro_tumor_test_list = ['004', '008', '010', '011', '013', '029', '033', '038', '046',
                             '048', '052', '064', '065', '066', '069', '074', '079', '084',
                             '092', '097', '099', '102', '108', '110', '116', '117', '122']

    test_list = ["test_" + str(t).zfill(3) + ".tif" for t in test_list]
    tumor_test_list = ["test_" + t + ".tif" for t in tumor_test_list]
    macro_tumor_test_list = ["test_" + t + ".tif" for t in macro_tumor_test_list]
    micro_tumor_test_list = ["test_" + t + ".tif" for t in micro_tumor_test_list]
    normal_test_list = list(set(test_list) - set(tumor_test_list))
    # ========================================================================================================
    stain_vector_pool = []
    with open("code/stain_vectors.json", 'r') as f:
        stainVectorDicts = json.load(f)
        for d in stainVectorDicts:
            if d["filename"] in train_list:
                stain_vector_pool.append(np.array(d["stain_matrix"]))
    print(f"stain_vector_pool created: {len(stain_vector_pool)} stain vectors.")
    # ========================================================================================================
    region_size = 64  # active learning selected region size at level 7
    n_query = 3  # active learning selected region amount

    # sampling_strategy = "full"
    # sampling_strategy = "random"
    sampling_strategy = "uncertainty_standard"
    # sampling_strategy = "uncertainty_non_square"
    # sampling_strategy = "uncertainty_adapt"

    # initial_sampling_strategy = "non-random"
    initial_sampling_strategy = "random"

    res_path = "exp_" + str(seed) + "_" + str(exp_id) + "_" + sampling_strategy + \
               "_n_query_" + str(n_query) + "_region_size_" + str(region_size)
    log_file = os.path.join(res_path, res_path + '.txt')


def Log(msg, filename):
    print(msg)
    date = datetime.now(timezone('Europe/Berlin')).strftime("%Y_%m_%d_%H_%M_%S")
    log_file = open(filename, "a")
    log_file.write(date + ": " + msg + "\n")
    log_file.close()
