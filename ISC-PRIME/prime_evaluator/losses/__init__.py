
def get_loss_by_name(loss_name, args):
    if loss_name.lower() == 'pred_classify':
        from .loss_funcs import PredClassify
        return PredClassify(args)
    elif loss_name.lower() == 'score_classify':
        from .loss_funcs import ScoreClassify
        return ScoreClassify(args)

    elif loss_name.lower() == 'dual_score_classify':
        from .loss_funcs import DualClassify
        return DualClassify(args, type='score')

    elif loss_name.lower() == 'pred_regress':
        from .loss_funcs import PredRegress
        return PredRegress(args)

    else:
        assert False, "Unknown loss type"
