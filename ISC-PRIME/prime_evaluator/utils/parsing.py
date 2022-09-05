import os
import argparse


def parse_arguments():
    """Parse Arguments."""
    parser = argparse.ArgumentParser()

    ############################ Dataset ############################
    parser.add_argument("--flag_dataset", type=str, default="centerline_sd",
                        help="dataset type (centerline_xy / centerline_sd)")
    parser.add_argument("--dataset", type=str, default="/home/joe/ServerBackup/RulePRIMEV2/compute_feature/feature/frenet_val/split",
                        help="path to the file which has train features.")
    parser.add_argument("--dataset_val", type=str, default="/home/joe/ServerBackup/RulePRIMEV2/compute_feature/feature/frenet_val/split",
                        help="path to the file which has validation features.")
    parser.add_argument("--data_root", type=str, default="/home/joe/ServerBackup/final_version_rule_equal_interval_0_25",
                        help="path to the file which has vector dataloader.")
    parser.add_argument("--data_augment", action="store_true", default=True,
                        help="(only acts in training dataset) If set, randomly scaling the sequence in a range")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers used for dataloader")

    ############################ Network ############################
    parser.add_argument("--net_name", type=str, default="traj_classifier",
                        help="Network type to be specified (traj_classifier / lane_classifier, vector_net_classifier)")
    parser.add_argument("--mode", type=str, default="train",
                        help="train/finetune/val/submit")
    parser.add_argument("--loss_name", type=str, default="score_classify",
                        help="Loss type to be specified (pred_classify / pred_regress / dual_score_classify / score_classify)")
    parser.add_argument("--dist_metric", type=str, default="byAvg",
                        help="dataset type (byMax / byAvg / byEndpt / by2pt / by3pt)")
    parser.add_argument("--dist_weighted", action="store_true", default=False,
                        help="If set, use weighted distance in finding oracle and scoring predictions")
    parser.add_argument("--score_temp", type=float, default=0.1,
                        help="temperature term used for calculating scores of all given predicted trajs (FOR score_classify loss)")
    parser.add_argument("--optim", type=str, default="adam", help="type of optimizer to be used (sgd / adam)")

    ##### Parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size used in training")
    parser.add_argument("--num_epochs", type=int, default=80, help="Epoch number used in training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1, help="Decay rate")
    parser.add_argument('--lr_decay_epochs', type=int, default=[], nargs='+', help="Decay learning rate by given factor at the given epochs")

    ##### Save and Load
    parser.add_argument("--ckptdir", type=str, default="./results/rule/", help="default checkpoint directory")
    parser.add_argument("--subdir", type=str, default="", help="If not set, then save all files in {ckptdir/mode}")
    parser.add_argument("--path_weight", type=str,
                        # default="/home/joe/Desktop/trained_model/prime_output/rule_prime/model_best.pth",
                        help="location of the pretrained weights")
    parser.add_argument("--plot", type=str, default=None, help="If set, set plot mode to (fde / all / None)")
    parser.add_argument("--save", action="store_true", default=False, help="If set, save all the results in val mode")
    parser.add_argument("--print_every_n_steps", type=int, default=1500, help="print loss every n steps")
    parser.add_argument("--val_every_n_epoch", type=int, default=1, help="validate every n epoch")

    ############################ Nearly Unchanged ############################
    parser.add_argument("--obs_len", type=int, default=10, help="Observed length of the trajectory")
    parser.add_argument("--pred_len", type=int, default=30, help="Prediction Horizon")
    parser.add_argument("--use_cuda", action="store_true", default=True, help="If set, use cuda.")
    parser.add_argument("--rank", type=int, default=0, help="If set, use cuda.")
    parser.add_argument("--logging", type=str, default="info", help="Specify logging level")
    parser.add_argument("--viz_name", type=str, default="argo", help="Viz type to be specified.")
    parser.add_argument("--tensorboard", action="store_true", default=True, help="tensorboard is used")
    parser.add_argument("--tensorboard_dir", type=str, default="",  help="tensorboard info directory")

    return parser.parse_args()