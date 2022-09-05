
def get_model_by_name(net_name: str, args, device=None):
    if net_name.lower() == 'traj_classifier':
        from .traj_classifier import TrajClassifier
        return TrajClassifier(args)
    elif net_name.lower() == 'lane_classifier':
        from .lane_classifier import LaneTrajClassifier
        return LaneTrajClassifier(args)
    elif net_name.lower() == 'vector_net_classifier':
        from .vectornet_traj_classifier import TrajClassifier
        return TrajClassifier(args, device)
    else:
        assert False, "Unsupported model type"
