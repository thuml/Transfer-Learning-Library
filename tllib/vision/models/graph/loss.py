import torch
import torch.nn as nn

def reduced_bce_logit_loss(y_pred, y_target):
    """
    Every item of y_target has n elements which may be labeled by nan.
    Nan values should not be used while calculating loss.
    So extract elements which are not nan first, and then calculate loss.
    """
    loss = nn.BCEWithLogitsLoss(reduction='none').cuda()
    is_labeled = ~torch.isnan(y_target)
    y_pred = y_pred[is_labeled].float()
    y_target = y_target[is_labeled].float()
    metrics = loss(y_pred, y_target)
    return metrics.mean()