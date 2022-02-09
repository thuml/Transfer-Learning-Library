import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge Distillation Loss.

    Args:
        T (double): Temperature. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'batchmean'``

    Inputs:
        - y_student (tensor): logits output of the student
        - y_teacher (tensor): logits output of the teacher

    Shape:
        - y_student: (minibatch, `num_classes`)
        - y_teacher: (minibatch, `num_classes`)

    """
    def __init__(self, T=1., reduction='batchmean'):
        super(KnowledgeDistillationLoss, self).__init__()
        self.T = T
        self.kl = nn.KLDivLoss(reduction=reduction)

    def forward(self, y_student, y_teacher):
        """"""
        return self.kl(F.log_softmax(y_student / self.T, dim=-1), F.softmax(y_teacher / self.T, dim=-1))
