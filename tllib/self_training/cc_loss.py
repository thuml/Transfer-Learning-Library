"""
@author: Ying Jin
@contact: sherryying003@gmail.com
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from tllib.modules.classifier import Classifier as ClassifierBase
from ..modules.entropy import entropy


__all__ = ['CCConsistency']


class CCConsistency(nn.Module):
    r"""
    CC Loss attach class confusion consistency to MCC.

    Args:
        temperature (float) : The temperature for rescaling, the prediction will shrink to vanilla softmax if
          temperature is 1.0.
        thr (float): The confidence threshold.

    .. note::
        Make sure that temperature is larger than 0. Confidence threshold is larger than 0, smaller than 1.0.

    Inputs: g_t
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`
        - g_t_strong (tensor): unnormalized classifier predictions on target domain, with strong data augmentation, :math:`g^t_{strong}`

    Shape:
        - g_t, g_t_strong: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.

    Examples::
        >>> temperature = 2.0
        >>> loss = CCConsistency(temperature)
        >>> # logits output from target domain
        >>> g_t = torch.randn(batch_size, num_classes)
        >>> g_t_strong = torch.randn(batch_size, num_classes)
        >>> output = loss(g_t, g_t_strong)
    """

    def __init__(self, temperature: float, thr=0.7):
        super(CCConsistency, self).__init__()
        self.temperature = temperature
        self.thr = thr

    def forward(self, logits: torch.Tensor, logits_strong: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        logits = logits.detach()

        prediction_thr = F.softmax(logits / self.temperature, dim=1)
        max_probs, max_idx = torch.max(prediction_thr, dim=-1)
        mask_binary = max_probs.ge(self.thr)  ### 0.7 for DomainNet, 0.95 for other datasets
        mask = mask_binary.float().detach()

        if mask.sum() == 0:
            return 0, 0
        else:
            logits = logits[mask_binary]
            logits_strong = logits_strong[mask_binary]

            predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
            entropy_weight = entropy(predictions).detach()
            entropy_weight = 1 + torch.exp(-entropy_weight)
            entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)   # batch_size x 1
            class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
            class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)

            predictions_stong = F.softmax(logits_strong / self.temperature, dim=1)
            entropy_weight_strong = entropy(predictions_stong).detach()
            entropy_weight_strong = 1 + torch.exp(-entropy_weight_strong)
            entropy_weight_strong = (batch_size * entropy_weight_strong / torch.sum(entropy_weight_strong)).unsqueeze(dim=1)   # batch_size x 1
            class_confusion_matrix_strong = torch.mm((predictions_stong * entropy_weight_strong).transpose(1, 0), predictions_stong)  # num_classes x num_classes
            class_confusion_matrix_strong = class_confusion_matrix_strong / torch.sum(class_confusion_matrix_strong, dim=1)

            consistency_loss = ((class_confusion_matrix - class_confusion_matrix_strong) ** 2).sum()  / num_classes * mask.sum() / batch_size
            #mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
            return consistency_loss, mask.sum()/batch_size