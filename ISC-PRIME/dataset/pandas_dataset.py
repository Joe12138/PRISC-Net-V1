import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_DICT = {
    "case_id": 0,
    "track_id": 1,
    "frame_id": 2,
    "timestamp_ms": 3,
    "agent_type": 4,
    "x": 5,
    "y": 6,
    "vx": 7,
    "vy": 8,
    "psi_rad": 9,
    "length": 10,
    "width": 11
}

DATA_TYPE_DICT = {
    "case_id": int,
    "track_id": int,
    "frame_id": int,
    "timestamp_ms": int,
    "agent_type": str,
    "x": float,
    "y": float,
    "vx": float,
    "vy": float,
    "psi_rad": float,
    "length": float,
    "width": float
}


class DatasetPandas:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_df = pd.read_csv(self.data_path, dtype=DATA_TYPE_DICT)

    def get_case_data(self, case_id: int) -> pd.DataFrame:
        return self.data_df[self.data_df["case_id"] == case_id]

    def get_track_data(self, case_id: int, track_id: int) -> np.ndarray:
        case_df = self.data_df[self.data_df["case_id"] == case_id]
        return case_df[case_df["track_id"] == track_id].values


if __name__ == '__main__':
    import sys
    sys.path.append("/home/joe/Desktop/TRCVTPP/RulePRIMEV2")
    from hdmap.visual.map_vis import draw_lanelet_map
    from hdmap.hd_map import HDMap

    mode = "train"
    data_p = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"+mode+"/DR_USA_Intersection_EP1_"+mode+".csv"
    map_p = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/maps/DR_USA_Intersection_EP1.osm"

    pandas_data = DatasetPandas(data_path=data_p)
    case_set = pandas_data.data_df[["case_id"]].drop_duplicates(["case_id"]).values
    hd_map = HDMap(osm_file_path=map_p)

    for i in range(case_set.shape[0]):
        case_id = int(case_set[i])
        case_df = pandas_data.data_df[pandas_data.data_df["case_id"] == case_id]

        track_set = case_df[["track_id"]].drop_duplicates(["track_id"]).values
        axes = plt.subplot(111)
        axes = draw_lanelet_map(hd_map.lanelet_map, axes)
        for j in range(track_set.shape[0]):
            track_id = int(track_set[j])

            # print(case_id, track_id)
            track_full_info = case_df[case_df["track_id"] == track_id].values

            if track_full_info[0, [DATA_DICT["agent_type"]]][0] != "car":
                continue
            traj_array = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")

            # axes.plot(traj_array[:10, 0], traj_array[:10, 1], color="red")
            if case_id == 1 and track_id == 14:
                axes.plot(traj_array[:, 0], traj_array[:, 1], color="red", linewidth=2)
                axes.scatter(traj_array[-1, 0], traj_array[-1, 1], color="red", s=25)
            else:
                axes.plot(traj_array[:, 0], traj_array[:, 1], color="purple", linewidth=1)
                axes.scatter(traj_array[-1, 0], traj_array[-1, 1], color="purple", s=15)
            # axes.text(traj_array[-1, 0]-0.5, traj_array[-1, 1]-0.5, str(track_id)+"_"+str(case_id))

        if i % 10 == 0:
            plt.show()
            plt.cla()
            plt.clf()

            break