import torch


class EqualWeightedCombiner:
    """
    Equal Weight optimizer.

    Attributes:
        task_names (List[str]): List of task names.
        device (str): Device to perform computations on.

    """

    def __init__(self, task_names, device):
        self.task_names = task_names
        self.device = device

    def combine(self, per_task_grad):
        """
        Perform Equal Weight optimization on the given gradients.

        Args:
            per_task_grad (List[torch.Tensor]): List of gradients for each task.

        Returns:
            torch.Tensor: Adjusted gradient.
        """
        per_task_grad = torch.stack(per_task_grad)
        adjusted_grad = per_task_grad.sum(0)
        return adjusted_grad