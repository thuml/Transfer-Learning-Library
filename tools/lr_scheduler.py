

class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer, init_lr=0.01, gamma=0.001, decay_rate=0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1

