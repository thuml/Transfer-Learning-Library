"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, List, Tuple

from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F


class AutomaticUpdateClassWeightModule(object):
    r"""
    Calculating class weight based on the output of classifier. See ``ClassWeightModule`` about the details of the calculation.
    Every N iterations, the class weight is updated automatically.

    Args:
        update_steps (int): N, the number of iterations to update class weight.
        data_loader (torch.utils.data.DataLoader): The data loader from which we can collect classification outputs.
        classifier (torch.nn.Module): Classifier.
        num_classes (int): Number of classes.
        device (torch.device): The device to run classifier.
        temperature (float, optional): T, temperature in ClassWeightModule. Default: 0.1
        partial_classes_index (list[int], optional): The index of partial classes. Note that this parameter is \
          just for debugging, since in real-world dataset, we have no access to the index of partial classes. \
          Default: None.

    Examples::

        >>> class_weight_module = AutomaticUpdateClassWeightModule(update_steps=500, ...)
        >>> num_iterations = 10000
        >>> for _ in range(num_iterations):
        >>>     class_weight_module.step()
        >>>     # weight for F.cross_entropy
        >>>     w_c = class_weight_module.get_class_weight_for_cross_entropy_loss()
        >>>     # weight for dalib.addaptation.dann.DomainAdversarialLoss
        >>>     w_s, w_t = class_weight_module.get_class_weight_for_adversarial_loss()
    """

    def __init__(self, update_steps: int, data_loader: DataLoader,
                 classifier: nn.Module, num_classes: int,
                 device: torch.device, temperature: Optional[float] = 0.1,
                 partial_classes_index: Optional[List[int]] = None):
        self.update_steps = update_steps
        self.data_loader = data_loader
        self.classifier = classifier
        self.device = device
        self.class_weight_module = ClassWeightModule(temperature)
        self.class_weight = torch.ones(num_classes).to(device)
        self.num_steps = 0
        self.partial_classes_index = partial_classes_index
        if partial_classes_index is not None:
            self.non_partial_classes_index = [c for c in range(num_classes) if c not in partial_classes_index]

    def step(self):
        self.num_steps += 1
        if self.num_steps % self.update_steps == 0:
            all_outputs = collect_classification_results(self.data_loader, self.classifier, self.device)
            self.class_weight = self.class_weight_module(all_outputs)

    def get_class_weight_for_cross_entropy_loss(self):
        """
        Outputs: weight for F.cross_entropy

        Shape: :math:`(C, )` where C means the number of classes.
        """
        return self.class_weight

    def get_class_weight_for_adversarial_loss(self, source_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Outputs:
            - w_s: source weight for :py:class:`~dalib.adaptation.dann.DomainAdversarialLoss`
            - w_t: target weight for :py:class:`~dalib.adaptation.dann.DomainAdversarialLoss`

        Shape:
            - w_s: :math:`(minibatch, )`
            - w_t: :math:`(minibatch, )`
        """
        class_weight_adv_source = self.class_weight[source_labels]
        class_weight_adv_target = torch.ones_like(class_weight_adv_source) * class_weight_adv_source.mean()
        return class_weight_adv_source, class_weight_adv_target

    def get_partial_classes_weight(self):
        """
        Get class weight averaged on the partial classes and non-partial classes respectively.

        .. warning::

            This function is just for debugging, since in real-world dataset, we have no access to the index of \
            partial classes and this function will throw an error when `partial_classes_index` is None.
        """
        assert self.partial_classes_index is not None
        return torch.mean(self.class_weight[self.partial_classes_index]), torch.mean(
            self.class_weight[self.non_partial_classes_index])


class ClassWeightModule(nn.Module):
    r"""
    Calculating class weight based on the output of classifier.
    Introduced by `Partial Adversarial Domain Adaptation (ECCV 2018) <https://arxiv.org/abs/1808.04205>`_

    Given classification logits outputs :math:`\{\hat{y}_i\}_{i=1}^n`, where :math:`n` is the dataset size,
    the weight indicating the contribution of each class to the training can be calculated as
    follows

    .. math::
        \mathcal{\gamma} = \dfrac{1}{n} \sum_{i=1}^{n}\text{softmax}( \hat{y}_i / T),

    where :math:`\mathcal{\gamma}` is a :math:`|\mathcal{C}|`-dimensional weight vector quantifying the contribution
    of each class and T is a hyper-parameters called temperature.

    In practice, it's possible that some of the weights are very small, thus, we normalize weight :math:`\mathcal{\gamma}`
    by dividing its largest element, i.e. :math:`\mathcal{\gamma} \leftarrow \mathcal{\gamma} / max(\mathcal{\gamma})`

    Args:
        temperature (float, optional): hyper-parameters :math:`T`. Default: 0.1

    Shape:
        - Inputs: (minibatch, :math:`|\mathcal{C}|`)
        - Outputs: (:math:`|\mathcal{C}|`,)
    """

    def __init__(self, temperature: Optional[float] = 0.1):
        super(ClassWeightModule, self).__init__()
        self.temperature = temperature

    def forward(self, outputs: torch.Tensor):
        outputs.detach_()
        softmax_outputs = F.softmax(outputs / self.temperature, dim=1)
        class_weight = torch.mean(softmax_outputs, dim=0)
        class_weight = class_weight / torch.max(class_weight)
        class_weight = class_weight.view(-1)
        return class_weight


def collect_classification_results(data_loader: DataLoader, classifier: nn.Module,
                                   device: torch.device) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `classifier` to collect classification results

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        classifier (torch.nn.Module): A classifier.
        device (torch.device)

    Returns:
        Classification results in shape (len(data_loader), :math:`|\mathcal{C}|`).
    """
    training = classifier.training
    classifier.eval()
    all_outputs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.to(device)
            output = classifier(images)
            all_outputs.append(output)
    classifier.train(training)
    return torch.cat(all_outputs, dim=0)
