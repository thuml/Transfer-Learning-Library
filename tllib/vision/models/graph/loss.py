import torch
import torch.nn as nn

def reduced_bce_logit_loss(y_pred, y_target):
    """
    Compute loss for obg-colpcba dataset. 
    There are 128 tasks for ogb-colpcba which predict the presence or
    absence of 128 different kinds of biological activities.
    y_target could contain NaN values, indicating that the corresponding
    biological assays were not performed on the given molecule
    """
    loss = nn.BCEWithLogitsLoss(reduction='none').cuda()
    is_labeled = ~torch.isnan(y_target)
    y_pred = y_pred[is_labeled].float()
    y_target = y_target[is_labeled].float()
    metrics = loss(y_pred, y_target)
    return metrics.mean()