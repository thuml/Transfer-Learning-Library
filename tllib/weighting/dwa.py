import torch
import torch.nn.functional as F


class DynamicWeightAverageLossCombiner:
    """
        A class that combines losses from multiple tasks using dynamic weighted averaging.

        Args:
            task_names (list): A list of task names.
            num_epochs (int): The total number of epochs.
            temperature (float): The temperature parameter for softmax weighting.
            device (str): The device to be used for computations.
    """

    def __init__(self, task_names, num_epochs, temperature, device):
        self.task_names = task_names
        self.device = device
        self.loss_history = torch.zeros(num_epochs, len(task_names)).to(device)
        self.temperature = temperature

    def combine(self, per_task_loss, epoch):
        """
        Combines the losses from multiple tasks using dynamic weighted averaging.

        Args:
            per_task_loss (list): A list of losses from each task.
            epoch (int): The current epoch.

        Returns:
            torch.Tensor: The combined loss.

        """
        if epoch > 1:
            weights = self.loss_history[epoch - 1, :] / self.loss_history[epoch - 2, :]
            weights = F.softmax(weights / self.temperature, dim=-1)
        else:
            weights = torch.ones(len(self.task_names)).to(self.device) / len(self.task_names)
        return sum([loss * weight for loss, weight in zip(per_task_loss, weights)])

    def update(self, per_task_loss, epoch):
        """
        Updates the loss history with the losses from each task at a given epoch.

        Args:
            per_task_loss (list): A list of losses from each task.
            epoch (int): The current epoch.

        """
        for task_idx, task_loss in enumerate(per_task_loss):
            self.loss_history[epoch, task_idx] = task_loss