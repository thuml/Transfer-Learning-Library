import torch


class AutomaticUpdateDomainWeightModule(object):

    def __init__(self, num_domains: int, eta: float, device):
        self.domain_weight = torch.ones(num_domains).to(device) / num_domains
        self.eta = eta

    def get_domain_weight(self):
        return self.domain_weight

    def update(self, loss_distribution: torch.Tensor):
        self.domain_weight *= (self.eta * loss_distribution.detach()).exp()
        self.domain_weight /= (self.domain_weight.sum())
