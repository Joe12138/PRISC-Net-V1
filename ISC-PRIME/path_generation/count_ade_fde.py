import os 
import pickle
import numpy as np

metrics_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/seg_file/val_metrics"

metrics_list = os.listdir(metrics_prefix)

ade_list = []
fde_list = []

for file_name in metrics_list:
    with open(os.path.join(metrics_prefix, file_name), "rb") as f:
        ade_fde_dict = pickle.load(f)
        
    ade_list.extend(ade_fde_dict["ade"])
    fde_list.extend(ade_fde_dict["fde"])
    
print("ade: avg={}, std={}".format(np.mean(ade_list), np.std(ade_list)))
print("fde: avg={}, std={}".format(np.mean(fde_list), np.std(fde_list)))
