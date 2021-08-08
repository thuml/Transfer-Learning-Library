import torch


class AutomaticUpdateDomainWeightModule(object):
    r"""
    Maintaining group weight based on loss history of all domains according
    to `Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case
    Generalization (ICLR 2020) <https://arxiv.org/pdf/1911.08731.pdf>`_.

    Suppose we have :math:`N` domains. During each iteration, we first calculate unweighted loss among all
    domains, resulting in :math:`loss\in R^N`. Then we update domain weight by

    .. math::
        w_k = w_k * exp(loss_k ^{\eta}), \forall k \in [1, N]

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

    def get_domain_weight(self):
        """
        Outputs: domain weight for calculating final objective

        Shape: :math:`(N, )` where N means the number of domains
        """
        return self.domain_weight

    def update(self, loss_distribution: torch.Tensor):
        """
        Inputs:
            - loss_distribution (tensor): loss distribution among domains
        Shape:
            - loss_distribution: :math:`(N, )` where N means the number of domains
        """
        self.domain_weight *= (self.eta * loss_distribution.detach()).exp()
        self.domain_weight /= (self.domain_weight.sum())
