import os
import numpy as np
import math
import sys

sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

from tqdm import tqdm

from dataset.pandas_dataset import DatasetPandas, DATA_DICT

mode = "val"
all_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"

data_path = os.path.join(all_prefix, mode)
target_veh_path = os.path.join(all_prefix, f"{mode}_target_filter")

file_list = os.listdir(target_veh_path)

