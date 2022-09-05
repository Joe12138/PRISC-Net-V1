
def get_viz_by_name(viz_name, args):
    if viz_name.lower() == 'argo':
        from .argo_vis import ArgoVis
        return ArgoVis(args)
    else:
        return None