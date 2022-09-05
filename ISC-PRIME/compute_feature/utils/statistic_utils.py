from typing import List, Tuple, Union
from scipy.spatial.distance import cdist

import numpy as np
import pandas as pd
import pathlib
import os
import csv
import argparse
import pickle

class PredsErrorStat:
    '''
    Created to record sampling failures
    '''
    def __init__(self):
        self.seq_id = None
        self.min_ade = 0
        self.argo_min_ade = 0
        self.min_fde = 0

        self.lane_num = 0
        self.trajs_list = 0
        self.trajs_num = 0

        self.aux_ade = 0   # auxiliary work
        self.aux_fde = 0

        self.vel = 0
        self.acc = 0
        self.yaw = 0

    def add_veh_stat(self, veh_stat=None):
        # Other information want to store
        if veh_stat is None:
            veh_stat = [0.0, 0.0, 0.0]
        self.vel = veh_stat[0]
        self.acc = veh_stat[1]
        self.yaw = veh_stat[2]

    def add_seq_id(self, seq = 0):
        self.seq_id = seq


def calcu_preditions_error(multimodal_preds_xy: List[List[np.ndarray]],
                           track_gt_xy: Union[np.ndarray, None])-> PredsErrorStat:
    """
    Return a PredsErrorStat from multimodal prediction results and ground truth result
    """
    stat = PredsErrorStat()
    stat.lane_num = len(multimodal_preds_xy)
    stat.trajs_list = [len(trajs) for trajs in multimodal_preds_xy]
    stat.trajs_num = sum(stat.trajs_list)

    if track_gt_xy is None: # Under test mode, no ade/fde results
        stat.min_ade = stat.argo_min_ade = stat.min_fde = stat.aux_ade = stat.aux_fde = 0.0
    else:
        if stat.trajs_num > 0:
            # First trajs
            first_traj_of_lane = [[trajs[0]] for trajs in multimodal_preds_xy]
            first_trajs = np.vstack([np.array(lane_trajs) for lane_trajs in first_traj_of_lane if lane_trajs])
            stat.aux_fde, stat.aux_ade, _ = calcu_distance_error(first_trajs, track_gt_xy)
            # All trajs
            trajs = np.vstack([np.array(lane_trajs) for lane_trajs in multimodal_preds_xy if lane_trajs])
            stat.min_fde, stat.min_ade, stat.argo_min_ade = calcu_distance_error(trajs, track_gt_xy)
        else:
            stat.min_ade = stat.argo_min_ade = stat.min_fde = stat.aux_ade = stat.aux_fde = 0.0
    return stat


def calcu_distance_error(preds: np.ndarray, gt: np.ndarray)-> List:

    diff = preds - gt[np.newaxis, :, :]
    disp_error = np.hypot(diff[:, :, 0], diff[:, :, 1])
    min_fde = disp_error[:, -1].min()
    idx = np.argmin(disp_error[:, -1])
    min_ade = np.mean(disp_error, axis=1).min()
    argo_min_ade = np.mean(disp_error[idx])
    return [min_fde, min_ade, argo_min_ade]


def save_csv_stat(first_row, save_loc, stat: PredsErrorStat):

    # First write down columns names
    if first_row:
        with open(save_loc, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(stat.__dict__.keys())
            writer.writerow(stat.__dict__.values())
    else:
        with open(save_loc, 'a+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(stat.__dict__.values())


def export_error_case(file_name):
    error_case = []
    csvFile = open(file_name, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        # trajs_num sixth one
        if float(item[6]) == 0.0:
            error_case.append(item[0])
    csvFile.close()

    dataframe = pd.DataFrame({'seq': error_case})

    return dataframe

def merge_csv(csv_save_dir):
    csv_files = os.listdir(csv_save_dir)
    # Sorted by number
    csv_files.sort()
    csv_idx = []

    for idx, csv_file in enumerate(csv_files):
        # MARK: If exists output.csv, delete
        if csv_file == 'output.csv':
            os.remove(csv_save_dir+'/'+'output.csv')
            continue
        if csv_file.endswith(".csv"):
            csv_idx.append(idx)
    csv_files = [csv_files[idx] for idx in csv_idx]
    df1 = pd.read_csv(csv_save_dir+'/'+csv_files[0], encoding='gbk')

    for file in csv_files:
        df2 = pd.read_csv(csv_save_dir+'/'+file, encoding='gbk')
        df1 = pd.concat([df1, df2], axis=0, ignore_index=True)

    # Remove duplicate lines and save
    df1 = df1.drop_duplicates()
    df1.to_csv(csv_save_dir+'/'+'output.csv', index=False)

def clean_error_data(data, data_value):
    error_idx_list = []
    for idx, error_case in enumerate(data):
        if error_case == data_value:
            error_idx_list.append(idx)

    return error_idx_list

def find_special_value(dataframe, data, start_value, end_value):
    idx_list = []
    for idx, special_value in enumerate(data):
        if special_value >= start_value and special_value <= end_value:
            idx_list.append(dataframe.seq_id[idx])

    return idx_list

def const_vel_correction(vehicle, seq, save_dir):
    yaw = vehicle.state.yaw
    vel = vehicle.state.vel

    yaw_list = [yaw+0.5, yaw+0.3, yaw+0.6, yaw-0.3, yaw+0.9, yaw]
    vel_list = [vel*1.5, vel*0.6, vel*1.0, vel*0.8, vel*0.8, vel*1.0]

    future_loc = []
    for i in range(6):
        future_loc.append(np.zeros((30, 2)))
    for i in range(6):
        yaw = yaw_list[i]
        vel = vel_list[i]
        x = vehicle.state.x
        y = vehicle.state.y
        for j in range(30):
            future_loc[i][j][0] = x + 0.1 * vel * np.cos(yaw)
            future_loc[i][j][1] = y + 0.1 * vel * np.sin(yaw)
            x = future_loc[i][j][0]
            y = future_loc[i][j][1]

    save_pkl = np.concatenate(future_loc, axis=0).reshape(6,30,2)
    end_pts = save_pkl[:,-1]
    print( cdist(end_pts, end_pts) > 3.0)

    fut_dict = {int(seq):save_pkl}
    pickle.dump(fut_dict, open(save_dir + f'/{seq}.pkl', 'wb'))
    agent_futs_xy = [future_loc]

    return agent_futs_xy


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./stat",
                        help="Directory where the traj error csv files are saved")

    return parser.parse_args()

def main():
    args = parse_arguments()
    root_dir = pathlib.Path(args.data_dir)
    abs_path = os.path.abspath(root_dir)
    merge_csv(args.data_dir)
    file_name = "{}/output.csv".format(abs_path)
    dataframe = export_error_case(file_name)
    dataframe.to_csv("{}/error_cases.csv".format(abs_path), index=False, sep=',')
    data = pd.read_csv(file_name, sep=',')
    error_idx_list = clean_error_data(data.trajs_num, 0.0)
    data_clean = data.drop(error_idx_list)

    # Find special value
    big_fde_list = find_special_value(data, data.trajs_num, start_value=1.0, end_value=20.0)
    # big_fde_list_2 = find_special_value(data, data.min_fde, 5.0, larger=True)
    # big_fde_list = list(set(big_fde_list_1).intersection(big_fde_list_2))
    missing_rate_list = find_special_value(data, data.min_fde, start_value=2.0, end_value=float("inf"))

    big_fde_dataframe = pd.DataFrame({'seq':big_fde_list})
    big_fde_dataframe.to_csv("{}/big_fde_case.csv".format(abs_path), index=False, sep=',')

    missing_rate = (len(missing_rate_list) + len(error_idx_list)) / len(data.seq_id)
    print(("==================== Predication Summary ====================\n" +
           "Tajectory Count:\t {} normal cases + {} bad cases\n".format(
               len(data_clean.min_ade), len(error_idx_list)) +
           "ADE Stat:\t        {:.2f} +- {:.2f} With min {:.2f} and max {:.2f} \n".format(
               data_clean.min_ade.mean(), data_clean.min_ade.std(), data_clean.min_ade.min(), data_clean.min_ade.max()) +
           "Argo ADE Stat:\t    {:.2f} +- {:.2f} With min {:.2f} and max {:.2f} \n".format(
               data_clean.argo_min_ade.mean(), data_clean.argo_min_ade.std(), data_clean.argo_min_ade.min(),
               data_clean.argo_min_ade.max()) +
           "FDE Stat:\t        {:.2f} +- {:.2f} With min {:.2f} and max {:.2f}\n".format(
               data_clean.min_fde.mean(), data_clean.min_fde.std(), data_clean.min_fde.min(), data_clean.min_fde.max()) +
           "================= First Trajectory Summary ===================\n"
           "1-Traj ADE Stat:\t{:.2f} +- {:.2f} With min {:.2f} and max {:.2f}\n".format(
               data_clean.aux_ade.mean(), data_clean.aux_ade.std(), data_clean.aux_ade.min(),
               data_clean.aux_ade.max()) +
           "1-Traj FDE Stat:\t{:.2f} +- {:.2f} With min {:.2f} and max {:.2f}\n\n".format(
               data_clean.aux_fde.mean(), data_clean.aux_fde.std(), data_clean.aux_fde.min(),
               data_clean.aux_fde.max())  +
           "Total missing rate of the scenarios is {:.4f}\n".format(missing_rate) +
           "================= Trajs and Lanes Summary ====================\n"
           "Lane Num :\t        {:.2f} +- {:.2f} With min {:.2f} and max {:.2f} \n".format(
               data_clean.lane_num.mean(), data_clean.lane_num.std(), data_clean.lane_num.min(), data_clean.lane_num.max()) +
           "Trajs Num:\t        {:.2f} +- {:.2f} With min {:.2f} and max {:.2f} \n".format(
               data_clean.trajs_num.mean(), data_clean.trajs_num.std(), data_clean.trajs_num.min(), data_clean.trajs_num.max()) +
           "==================== Vehicle Summary ====================\n"
           "ACC Stat:\t        {:.2f} +- {:.2f} With min {:.2f} and max {:.2f} \n".format(
               data_clean.acc.mean(), data_clean.acc.std(), data_clean.acc.min(), data_clean.acc.max()) +
           "VEL Stat:\t        {:.2f} +- {:.2f} With min {:.2f} and max {:.2f} \n".format(
               data_clean.vel.mean(), data_clean.vel.std(), data_clean.vel.min(), data_clean.vel.max()) +
           "YAW Stat:\t        {:.2f} +- {:.2f} With min {:.2f} and max {:.2f} \n".format(
               data_clean.yaw.mean(), data_clean.yaw.std(), data_clean.yaw.min(), data_clean.yaw.max())
           ))


if __name__ == '__main__':
    main()






