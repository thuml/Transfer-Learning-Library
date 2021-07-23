from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class TradeOffScheduler(object):

    def __init__(self, anneal_iters, trade_off, anneal_trade_off: Optional[float] = 1):
        self.anneal_trade_off = anneal_trade_off
        self.trade_off = trade_off
        self.count = 0
        self.anneal_iters = anneal_iters

    def step(self):
        self.count += 1

    def get_count(self):
        return self.count

    def get_trade_off(self):
        if self.count >= self.anneal_iters:
            return self.trade_off
        else:
            return self.anneal_trade_off


class IrmPenaltyLoss(nn.Module):

    def __init__(self):
        super(IrmPenaltyLoss, self).__init__()
        self.scale = torch.tensor(1.).requires_grad_()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_1 = F.cross_entropy(y[::2] * self.scale, labels[::2])
        loss_2 = F.cross_entropy(y[1::2] * self.scale, labels[1::2])
        grad_1 = autograd.grad(loss_1, [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [self.scale], create_graph=True)[0]
        penalty = torch.sum(grad_1 * grad_2)
        return penalty
