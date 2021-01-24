import torch.nn.functional as F
import math


def robust_entropy(y, ita=1.5, num_classes=19, reduction='mean'):
    P = F.softmax(y, dim=1)
    logP = F.log_softmax(y, dim=1)
    PlogP = P * logP
    ent = -1.0 * PlogP.sum(dim=1)
    ent = ent / math.log(num_classes)

    # compute robust entropy
    ent = ent ** 2.0 + 1e-8
    ent = ent ** ita

    if reduction == 'mean':
        return ent.mean()
    else:
        return ent