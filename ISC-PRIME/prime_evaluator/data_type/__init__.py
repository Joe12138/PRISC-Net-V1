def get_datatype_by_name(flag, dataset_loc, args, augment=False):

    if flag == 'centerline_xy':
        pass
    elif flag == 'centerline_sd':
        from .centerline_sd_dataset import CenterlineSDDataset
        return CenterlineSDDataset(dataset_loc, args.obs_len, args.obs_len + args.pred_len, [(0, 0)], augment, args)
    else:
        assert False, "Unsupported datatype flag"