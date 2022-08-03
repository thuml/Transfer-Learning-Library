"""
Modified from https://github.com/facebookresearch/DomainBed
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch


class AutomaticUpdateDomainWeightModule(object):
    r"""
    Maintaining group weight based on loss history of all domains according
    to `Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case
    Generalization (ICLR 2020) <https://arxiv.org/pdf/1911.08731.pdf>`_.

    Suppose we have :math:`N` domains. During each iteration, we first calculate unweighted loss among all
    domains, resulting in :math:`loss\in R^N`. Then we update domain weight by

    .. math::
        w_k = w_k * \text{exp}(loss_k ^{\eta}), \forall k \in [1, N]

    where :math:`\eta` is the hyper parameter which ensures smoother change of weight.
    As :math:`w \in R^N` denotes a distribution, we `normalize`
    :math:`w` by its sum. At last, weighted loss is calculated as our objective

    .. math::
        objective = \sum_{k=1}^N w_k * loss_k

    Args:
        num_domains (int): The number of source domains.
        eta (float): Hyper parameter eta.
        device (torch.device): The device to run on.
    """

    def __init__(self, num_domains: int, eta: float, device):
        self.domain_weight = torch.ones(num_domains).to(device) / num_domains
        self.eta = eta

    def get_domain_weight(self, sampled_domain_idxes):
        """Get domain weight to calculate final objective.

        Inputs:
            - sampled_domain_idxes (list): sampled domain indexes in current mini-batch

        Shape:
            - sampled_domain_idxes: :math:`(D, )` where D means the number of sampled domains in current mini-batch
            - Outputs: :math:`(D, )`
        """
        domain_weight = self.domain_weight[sampled_domain_idxes]
        domain_weight = domain_weight / domain_weight.sum()
        return domain_weight

    def update(self, sampled_domain_losses: torch.Tensor, sampled_domain_idxes):
        """Update domain weight using loss of current mini-batch.

        Inputs:
            - sampled_domain_losses (tensor): loss of among sampled domains in current mini-batch
            - sampled_domain_idxes (list): sampled domain indexes in current mini-batch

        Shape:
            - sampled_domain_losses: :math:`(D, )` where D means the number of sampled domains in current mini-batch
            - sampled_domain_idxes: :math:`(D, )`
        """
        sampled_domain_losses = sampled_domain_losses.detach()

        for loss, idx in zip(sampled_domain_losses, sampled_domain_idxes):
            self.domain_weight[idx] *= (self.eta * loss).exp()
