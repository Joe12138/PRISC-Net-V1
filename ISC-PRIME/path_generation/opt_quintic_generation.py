import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
import math
import copy
import numpy as np
import json
from path_generation.utils.polynomial_planner import QuinticPolynomial


def quintic_polynomial_planner(sx: float, sy: float, syaw: float, sv: float, sa: float,
                               gx: float, gy: float, gyaw: float, gv: float, ga: float,
                               dt: float = 0.1, T: float = 3):
    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)
    vxg = gv * math.cos(gyaw)
    vyg = gv * math.sin(gyaw)

    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)
    axg = ga * math.cos(gyaw)
    ayg = ga * math.sin(gyaw)

    xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
    yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)
    rx, ry = [], []

    for t in np.arange(dt, T + dt, dt):
        rx.append(xqp.calc_point(t))
        ry.append(yqp.calc_point(t))

    return rx, ry

def get_cost_info(sx: float, sy: float, sv: float, sa: float, pred_traj: np.ndarray):
    start_x, start_y = sx, sy
    start_v, start_a = sv, sa
    jeck_cost = 0
    for i in range(pred_traj.shape[0]):
        cur_p_x, cur_p_y = pred_traj[i][0], pred_traj[i][1]
        cur_v = math.sqrt((cur_p_x-start_x)**2+(cur_p_y-start_y)**2)/0.1
        cur_a = (cur_v-start_v)/0.1
        jeck_cost += ((cur_a-start_a)/0.1)**2
        
        start_x, start_y = cur_p_x, cur_p_y
        start_v, start_a = cur_v, cur_a
    
    vel_diff = (start_v-sv)**2
    lateral_diff = min(abs(start_x-sx), abs(start_y-sy))
    
    return jeck_cost, vel_diff, lateral_diff
        
    
def opt_planner(sx: float, sy: float, gx: float, gy: float, 
                sv: float, sa: float, syaw: float,
                min_v: float, max_v: float, min_a: float, max_a: float,
                min_yaw: float, max_yaw: float):
    jeck_coe = 0.1
    vel_diff_coe = 1
    lateral_coe = 1
    min_cost = 10e9
    best_traj = None
    for gv in np.arange(min_v, max_v, 0.3):
        for ga in np.arange(min_a, max_a, 0.2):
            for gyaw in np.arange(min_yaw, max_yaw, math.pi*10/180):
                rx, ry = quintic_polynomial_planner(
                    sx=sx,
                    sy=sy,
                    syaw=syaw,
                    sv=sv,
                    sa=sa,
                    gx=gx,
                    gy=gy,
                    gyaw=gyaw,
                    gv=gv,
                    ga=ga
                )
                
                pred_traj_array = np.array([rx, ry]).transpose()
                jeck_cost, vel_diff, lateral_diff = get_cost_info(
                    sx=sx,
                    sy=sy,
                    sv=sv,
                    sa=sa,
                    pred_traj=pred_traj_array
                )
                cur_cost = jeck_cost*jeck_coe+vel_diff*vel_diff_coe+lateral_diff*lateral_coe
                if cur_cost < min_cost:
                    min_cost = cur_cost
                    best_traj = copy.deepcopy(pred_traj_array)
    return best_traj


def get_target_veh_list(target_veh_path: str, file_name: str):
    target_veh_list = []
    scene_name = file_name[:-5]
    with open(os.path.join(target_veh_path, file_name), "r", encoding="UTF-8") as f:
        target_dict = json.load(f)

        for k in target_dict.keys():
            case_id = int(k)

            for track_id in target_dict[k]:
                target_veh_list.append((scene_name, case_id, track_id))

        f.close()
    return target_veh_list


if __name__ == '__main__':
    import os
    from tqdm import tqdm
    import pickle
    import matplotlib.pyplot as plt
    from dataset.pandas_dataset import DatasetPandas, DATA_DICT
    from util_dir.metric import get_ade, get_fde
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", type=str, default="seg_11.pkl")

    args = parser.parse_args()
    
    mode = "val"
    prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"
    data_path_prefix = f"{prefix}{mode}"
    target_veh_path = f"{prefix}{mode}_target_filter"
    
    data_file_list = os.listdir(data_path_prefix)
    data_dict = {}
    for data_file in data_file_list:
        data_pandas = DatasetPandas(data_path=os.path.join(data_path_prefix, data_file))
        scene_name = data_file[:-8]
        data_dict[scene_name] = data_pandas
        
    seg_file_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/seg_file/val"
    # seg_file_list = os.listdir(seg_file_prefix)
    # str_content = ""
    # for idx, seq_file in enumerate(seg_file_list):
    #     if idx == 0:
    #         str_content += f"python opt_quintic_generation.py -f {seq_file}"
    #     else:
    #         str_content += f" & python opt_quintic_generation.py -f {seq_file}"
    
    # print(str_content)
    # exit(0)
        
    with open(os.path.join(seg_file_prefix, args.file_name), "rb") as f:
        seg_dict = pickle.load(f)
    
    target_veh_list = list(seg_dict.values())
    
    ade_list = []
    fde_list = []
    
    save_dir = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/seg_file/val_metrics"
    for scene_name, case_id, track_id in tqdm(target_veh_list):
        dataset_pandas = data_dict[scene_name]
        track_full_info = dataset_pandas.get_track_data(case_id=case_id, track_id=track_id)

        track_full_xy = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
        track_full_v = track_full_info[:, [DATA_DICT["vx"], DATA_DICT["vy"]]].astype(float)
        track_full_yaw = track_full_info[:, [DATA_DICT["psi_rad"]]].astype(float)
        
        vxs, vys = track_full_v[9][0], track_full_v[9][1]
        vxn, vyn = track_full_v[10][0], track_full_v[10][1]

        vs = math.sqrt(vxs ** 2 + vys ** 2)
        vn = math.sqrt(vxn ** 2 + vyn ** 2)

        a_s = (vn - vs) / 0.1
        
        pred_traj = opt_planner(
            sx=track_full_xy[9][0],
            sy=track_full_xy[9][1],
            gx=track_full_xy[-1][0],
            gy=track_full_xy[-1][1],
            syaw=track_full_yaw[9][0],
            sv=vs,
            sa=a_s,
            min_v=max(0, vs-5),
            max_v=vs+5,
            min_a=max(-2, a_s-2),
            max_a=min(3, a_s+2),
            min_yaw=track_full_yaw[9][0]-math.pi/6,
            max_yaw=track_full_yaw[9][0]+math.pi/6
        )
        
        if pred_traj is None:
            continue
        ade = get_ade(pred_traj, track_full_xy[10:])
        fde = get_fde(pred_traj, track_full_xy[10:])
        
        ade_list.append(ade)
        fde_list.append(fde)
            
            # print("ade = {}, fde = {}".format(ade, fde))
    # print("ade avg={}, std={}, min={}, max={}".format(np.mean(ade_list), np.std(ade_list), np.min(ade_list), np.max(ade_list)))
    # print("fde avg={}, std={}, min={}, max={}".format(np.mean(fde_list), np.std(fde_list), np.min(fde_list), np.max(fde_list)))
    
    metric_dict = {"ade": ade_list, "fde": fde_list}
    
    with open(os.path.join(save_dir, args.file_name), "wb") as f:
        pickle.dump(metric_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    
