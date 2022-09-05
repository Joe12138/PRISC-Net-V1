import sys

from pip import main
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

import gurobipy
import math
import numpy as np
import json

from path_generation.utils.polynomial_planner import QuinticPolynomial

def opt_quintic_polynomial_planner(sx: float, sy: float, syaw: float, sv: float, sa: float,
                                   gx: float, gy: float, gyaw: float, max_ve: float,
                                   dt: float=0.1, time_horizon: float = 3):
    traj_model = gurobipy.Model()
    ve = traj_model.addVar(lb=0, ub=max_ve, vtype=gurobipy.GRB.CONTINUOUS, name="ve")
    ae = traj_model.addVar(lb=-3, ub=2, vtype=gurobipy.GRB.CONTINUOUS, name="ae")
    traj_model.update()
    print(ve.lb)
    
    xa0 = sx
    xa1 = sv*math.cos(syaw)
    xa2 = sa*math.cos(syaw)/2
    xa3_part = -(20*xa0 - 20*gx + 12*xa1*time_horizon + 6*xa2*time_horizon**2 )/(2*time_horizon**3)
    xa3_ve_coe = -(8*time_horizon*math.cos(gyaw))/(2*time_horizon**3)
    xa4_part = (15*xa0 - 15*gx + 8*xa1*time_horizon + 3*xa2*time_horizon**2)/time_horizon**4
    xa4_ve_coe = (7*time_horizon*math.cos(gyaw))/time_horizon**4
    xa5_part = -(12*xa0 - 12*gx + 6*xa1*time_horizon + 2*xa2*time_horizon**2)/(2*time_horizon**5)
    xa5_ve_coe = -(6*time_horizon*math.cos(gyaw))/(2*time_horizon**5)
    x_ae_coe = -(-time_horizon**2*math.cos(gyaw))/(2*time_horizon**3)
    
    ya0 = sy
    ya1 = sv*math.sin(syaw)
    ya2 = sa*math.sin(syaw)/2
    ya3_part = -(20*ya0 - 20*gy + 12*ya1*time_horizon + 6*ya2*time_horizon**2 )/(2*time_horizon**3)
    ya3_ve_coe = -(8*time_horizon*math.sin(gyaw))/(2*time_horizon**3)
    ya4_part = (15*ya0 - 15*gy + 8*ya1*time_horizon + 3*ya2*time_horizon**2)/time_horizon**4
    ya4_ve_coe = (7*time_horizon*math.sin(gyaw))/time_horizon**4
    ya5_part = -(12*ya0 - 12*gy + 6*ya1*time_horizon + 2*ya2*time_horizon**2)/(2*time_horizon**5)
    ya5_ve_coe = -(6*time_horizon*math.sin(gyaw))/(2*time_horizon**5)
    y_ae_coe = -(-time_horizon**2*math.sin(gyaw))/(2*time_horizon**3)
    
    obj_express = gurobipy.QuadExpr()
    start_x = sx
    start_y = sy
    for i in range(1, 31):
        t = i*0.1
        cur_x = xa0 + xa1*t + xa2*t**2 + (xa3_part+xa3_ve_coe*ve+x_ae_coe*ae)*t**3 + \
        (xa4_part+xa4_ve_coe*ve+x_ae_coe*ae)*t**4 + (xa5_part+xa5_ve_coe*ve+x_ae_coe*ae)*t**5
        cur_y = ya0 + ya1*t + ya2*t**2 + (ya3_part+ya3_ve_coe*ve+y_ae_coe*ae)*t**3 + \
        (ya4_part+ya4_ve_coe*ve+y_ae_coe*ae)*t**4 + (ya5_part+ya5_ve_coe*ve+y_ae_coe*ae)*t**5
        obj_express.add((cur_x-start_x)**2+(cur_y-start_y)**2)
        start_x = cur_x
        start_y = cur_y
        
    traj_model.setObjective(obj_express, sense=gurobipy.GRB.MINIMIZE)
    traj_model.optimize()
    
    # print(traj_model.status)
    
    # print("obj:", traj_model.objVal)
    # print(ve.x, ae.x)
    
    return ve.x, ae.x


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
    import matplotlib.pyplot as plt
    from dataset.pandas_dataset import DatasetPandas, DATA_DICT
    from path_generation.quintic_generation import quintic_polynomial_planner
    from util_dir.metric import get_ade, get_fde
    
    mode = "val"
    prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"
    data_path_prefix = f"{prefix}{mode}"

    target_veh_path = f"{prefix}{mode}_target_filter"

    scene_list = os.listdir(target_veh_path)
    ade_list = []
    fde_list = []
    
    for file_name in scene_list:
        print(file_name)
        dataset_pandas = DatasetPandas(data_path=os.path.join(data_path_prefix, f"{file_name[:-5]}_{mode}.csv"))
        target_veh_list = get_target_veh_list(target_veh_path=target_veh_path, file_name=file_name)
        
        for scene_name, case_id, track_id in tqdm(target_veh_list):
            track_full_info = dataset_pandas.get_track_data(case_id=case_id, track_id=track_id)

            track_full_xy = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
            track_full_v = track_full_info[:, [DATA_DICT["vx"], DATA_DICT["vy"]]].astype(float)
            track_full_yaw = track_full_info[:, [DATA_DICT["psi_rad"]]].astype(float)

            vxs, vys = track_full_v[9][0], track_full_v[9][1]
            vxn, vyn = track_full_v[10][0], track_full_v[10][1]

            vs = math.sqrt(vxs ** 2 + vys ** 2)
            vn = math.sqrt(vxn ** 2 + vyn ** 2)

            a_s = (vn - vs) / 0.1
            
            ve, ae = opt_quintic_polynomial_planner(
                sx=track_full_xy[9][0],
                sy=track_full_xy[9][1],
                syaw=track_full_yaw[9][0],
                sv=vs,
                sa=a_s,
                gx=track_full_xy[-1][0],
                gy=track_full_xy[-1][1],
                gyaw=track_full_yaw[9][0]+0.2,
                max_ve=7,
                time_horizon=3
            )
            
            rx, ry = quintic_polynomial_planner(
                sx=track_full_xy[9][0],
                sy=track_full_xy[9][1],
                syaw=track_full_yaw[9][0],
                sv=vs,
                sa=a_s,
                gx=track_full_xy[-1][0],
                gy=track_full_xy[-1][1],
                gyaw=track_full_yaw[9][0]+0.2,
                gv=ve,
                ga=ae
            )
            
            pred_traj = np.asarray([(x, y) for x, y in zip(rx, ry)])
            # plt.plot(track_full_xy[:10, 0], track_full_xy[:10, 1], color="purple", marker="o", zorder=10)
            # plt.plot(track_full_xy[10:, 0], track_full_xy[10:, 1], color="orange", marker="o", zorder=10)
            # plt.plot(rx, ry, color="blue", marker="s")
            # ax = plt.gca()
            # ax.set_aspect(1)
            # plt.show()
            # plt.cla()
            # plt.clf()
            # pred_traj_array = np.expand_dims(pred_traj, 0)
            ade = get_ade(pred_traj, track_full_xy[10:])
            fde = get_fde(pred_traj, track_full_xy[10:])
            
            ade_list.append(ade)
            fde_list.append(fde)
            
    print("ade avg={}, std={}, min={}, max={}".format(np.mean(ade_list), np.std(ade_list), np.min(ade_list), np.max(ade_list)))
    print("fde avg={}, std={}, min={}, max={}".format(np.mean(fde_list), np.std(fde_list), np.min(fde_list), np.max(fde_list)))