import torch
import random


class PCGradGradientWeightedCombiner:
    """
    PCGrad optimizer that adjusts gradients based on inner products between task gradients.

    Attributes:
        task_names (List[str]): List of task names.
        device (str): Device to perform computations on.

    """

    def __init__(self, task_names, device):
        self.task_names = task_names
        self.device = device

    def combine(self, per_task_grad):
        """
        Perform PCGrad optimization on the given gradients.

        Args:
            per_task_grad (List[torch.Tensor]): List of gradients for each task.

        Returns:
            torch.Tensor: Adjusted gradient.
        """
        per_task_grad = torch.stack(per_task_grad)
        weights = torch.ones(len(self.task_names)).to(self.device)

        adjusted_grad = per_task_grad.clone()
        for idx_i in range(len(self.task_names)):
            task_order = list(range(len(self.task_names)))
            random.shuffle(task_order)
            for idx_j in task_order:
                inner_product = torch.dot(adjusted_grad[idx_i], per_task_grad[idx_j])
                if inner_product < 0:
                    adjusted_grad[idx_i] -= inner_product * per_task_grad[idx_j] / (per_task_grad[idx_j].norm().pow(2))
                    weights[idx_j] -= (inner_product / (per_task_grad[idx_j].norm().pow(2)))
        adjusted_grad = adjusted_grad.sum(0)

        return adjusted_grad