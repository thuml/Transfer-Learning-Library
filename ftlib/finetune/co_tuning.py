from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression


__all__ = ['relationship_learning', 'direct_relationship_learning', 'get_feature', 'Classifier']


def calibrate(logits, labels):
    """
    calibrate by minimizing negative log likelihood.
    :param logits: pytorch tensor with shape of [N, N_c]
    :param labels: pytorch tensor of labels
    :return: float
    """
    scale = nn.Parameter(torch.ones(
        1, 1, dtype=torch.float32), requires_grad=True)
    optim = torch.optim.LBFGS([scale])

    def loss():
        optim.zero_grad()
        lo = nn.CrossEntropyLoss()(logits * scale, labels)
        lo.backward()
        return lo

    state = optim.state[scale]
    for i in range(20):
        optim.step(loss)
        print(f'calibrating, {scale.item()}')
        if state['n_iter'] < optim.state_dict()['param_groups'][0]['max_iter']:
            break

    return scale.item()


def softmax_np(x):
    max_el = np.max(x, axis=1, keepdims=True)
    x = x - max_el
    x = np.exp(x)
    s = np.sum(x, axis=1, keepdims=True)
    return x / s


def relationship_learning(train_logits, train_labels, validation_logits, validation_labels):
    """
    :param train_logits (pretrained logits): [N, N_p], where N_p is the number of classes in pre-trained dataset
    :param train_labels:  [N], where 0 <= each number < N_t, and N_t is the number of target dataset
    :param validation_logits (pretrained logits): [N, N_p]
    :param validation_labels:  [N]
    :return: [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
     """

    # convert logits to probabilities
    train_probabilities = softmax_np(train_logits)
    validation_probabilities = softmax_np(
        validation_logits)

    all_probabilities = np.concatenate(
        (train_probabilities, validation_probabilities))
    all_labels = np.concatenate((train_labels, validation_labels))

    Cs = []
    accs = []
    classifiers = []
    for C in [1e4, 3e3, 1e3, 3e2, 1e2, 3e1, 1e1, 3.0, 1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        cls = LogisticRegression(
            multi_class='multinomial', C=C, fit_intercept=False)
        cls.fit(train_probabilities, train_labels)
        val_predict = cls.predict(validation_probabilities)
        val_acc = np.sum((val_predict == validation_labels).astype(
            np.float)) / len(validation_labels)
        Cs.append(C)
        accs.append(val_acc)
        classifiers.append(cls)

    accs = np.asarray(accs)
    ind = int(np.argmax(accs))
    cls = classifiers[ind]
    del classifiers

    validation_logits = np.matmul(validation_probabilities, cls.coef_.T)
    validation_logits = torch.from_numpy(validation_logits.astype(np.float32))
    validation_labels = torch.from_numpy(validation_labels)

    scale = calibrate(validation_logits, validation_labels)

    p_target_given_pretrain = softmax_np(
        cls.coef_.T * scale)  # shape of [N_p, N_c], conditional probability p(target_class | pre-trained class)

    # in the paper, both ys marginal and yt marginal are computed
    # here we only use ys marginal to make sure p_pretrain_given_target is a valid conditional probability
    # (make sure p_pretrain_given_target[i] sums up to 1)
    pretrain_marginal = np.mean(all_probabilities, axis=0).reshape(
        (-1, 1))  # shape of [N_p, 1]
    p_joint_distribution = (p_target_given_pretrain * pretrain_marginal).T
    p_pretrain_given_target = p_joint_distribution / \
        np.sum(p_joint_distribution, axis=1, keepdims=True)

    return p_pretrain_given_target


def direct_relationship_learning(train_logits, train_labels, validation_logits, validation_labels):
    """
    The direct approach of learning category relationship.
    :param train_logits (pretrained logits): [N, N_p], where N_p is the number of classes in pre-trained dataset
    :param train_labels:  [N], where 0 <= each number < N_t, and N_t is the number of classes in target dataset
    :param validation_logits (pretrained logits): [N, N_p]
    :param validation_labels:  [N]
    :return: [N_c, N_p] matrix representing the conditional probability p(pre-trained class | target_class)
     """
    # convert logits to probabilities
    train_probabilities = softmax_np(train_logits)
    validation_probabilities = softmax_np(
        validation_logits)

    all_probabilities = np.concatenate(
        (train_probabilities, validation_probabilities))
    all_labels = np.concatenate((train_labels, validation_labels))

    N_t = np.max(all_labels) + 1 # the number of target classes
    conditional = []
    for i in range(N_t):
        this_class = all_probabilities[all_labels == i]
        average = np.mean(this_class, axis=0, keepdims=True)
        conditional.append(average)
    return np.concatenate(conditional)


def get_feature(loader, net):
    train_labels_list = []
    pretrained_labels_list = []

    for i, (train_inputs, train_labels) in enumerate(loader):
        net.eval()
        train_labels_list.append(train_labels)

        train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()
        pretrained_labels, _, _ = net(train_inputs)
        pretrained_labels = pretrained_labels.detach().cpu().numpy()

        pretrained_labels_list.append(pretrained_labels)

    all_train_labels = np.concatenate(train_labels_list, 0)
    all_pretrained_labels = np.concatenate(pretrained_labels_list, 0)
    return all_pretrained_labels, all_train_labels


class Classifier(nn.Module):
    """
    """

    def __init__(self, backbone: nn.Module, num_classes: int,  source_head: Optional[nn.Module] = None, finetune=True):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if source_head is not None:
            self.source_head = source_head

        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self._features_dim = self.backbone.out_features
        self.head = nn.Linear(self._features_dim, num_classes)
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = self.bottleneck(f)
        source_prediction = self.source_head(f)
        prediction = self.head(f)
        return source_prediction, prediction, f

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.source_head.parameters(), "lr": 0.1 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]
        return params
