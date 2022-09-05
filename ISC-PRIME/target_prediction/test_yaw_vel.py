# /home/joe/Desktop/TRCVTPP/RulePRIMEV2/target_prediction/target_output/yaw_output/06-09-10-05/checkpoint_iter85.ckpt

import os
import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
from os.path import join as pjoin
from datetime import datetime

import argparse

# from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader

from target_prediction.dataloader.graph_data import GraphData
from target_prediction.dataloader.target_interaction_loader import TargetInteractionInMem
from target_prediction.trainer.yaw_vel_trainer import YawVelTrainer


def test(args):
    """
    script to test the tnt model
    "param args:
    :return:
    """
    # config
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = pjoin(args.save_dir, time_stamp)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        raise Exception("The output folder does exists and is not empty! Check the folder.")
    else:
        os.makedirs(output_dir)

    # data loading
    try:
        test_set = TargetInteractionInMem(pjoin(args.data_root, "{}_intermediate".format(args.split)))
    except:
        raise Exception("Failed to load the data, please check the dataset!")

    # init trainer
    trainer = YawVelTrainer(
        trainset=test_set,
        evalset=test_set,
        testset=test_set,
        batch_size=args.batch_size,
        number_workers=args.num_workers,
        aux_loss=True,
        enable_log=False,
        with_cuda=args.with_cuda,
        cuda_device=args.cuda_device,
        save_folder=output_dir,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None,
        variable=args.pred_variable
    )

    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    variable = "yaw"
    parser.add_argument("-r", "--data_root", type=str,
                        default="/home/joe/ServerBackup/final_version_rule_equal_interval_0_25",
                        help="root dir for datasets")
    parser.add_argument("-s", "--split", type=str, default="val")

    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="number of batch_size")
    parser.add_argument("-w", "--num_workers", type=int, default=2,
                        help="dataloader worker size")
    parser.add_argument("-c", "--with_cuda", action="store_true", default=True,
                        help="training with CUDA: true, or false")
    parser.add_argument("-cd", "--cuda_device", type=int, default=0, nargs='+',
                        help="CUDA device ids")

    parser.add_argument("-rc", "--resume_checkpoint", type=str,
                        # default="/home/joe/Desktop/Rule-PRIME/Model/vx_output/06-28-11-25/checkpoint_iter141.ckpt",
                        help="resume a checkpoint for fine-tune")
    parser.add_argument("-rm", "--resume_model", type=str,
                        default="/home/joe/Desktop/Rule-PRIME/Model/yaw_output/06-27-18-51/best_YawVelPredict.pth",
                        help="resume a model state for fine-tune")

    parser.add_argument("-d", "--save_dir", type=str, default="test_result")
    parser.add_argument("-v", "--pred_variable", type=str, default=variable)
    args = parser.parse_args()
    test(args)
