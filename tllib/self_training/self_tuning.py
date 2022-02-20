"""
Adapted from https://github.com/thuml/Self-Tuning/tree/master
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from tllib.modules.classifier import Classifier as ClassifierBase


class Classifier(ClassifierBase):
    """Classifier class for Self-Tuning.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes.
        projection_dim (int, optional): Dimension of the projector head. Default: 128
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        In the training mode,
            - h: projections
            - y: classifier's predictions
        In the eval mode,
            - y: classifier's predictions

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - y: (minibatch, `num_classes`)
        - h: (minibatch, `projection_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, projection_dim=1024, bottleneck_dim=1024, finetune=True,
                 pool_layer=None):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        head = nn.Linear(1024, num_classes)
        super(Classifier, self).__init__(backbone, num_classes=num_classes, head=head, finetune=finetune,
                                         pool_layer=pool_layer, bottleneck=bottleneck, bottleneck_dim=bottleneck_dim)
        self.projector = nn.Linear(1024, projection_dim)
        self.projection_dim = projection_dim

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        # projections
        h = self.projector(f)
        h = normalize(h, dim=1)
        # predictions
        predictions = self.head(f)
        if self.training:
            return h, predictions
        else:
            return predictions

    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.projector.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
        ]

        return params


class SelfTuning(nn.Module):
    r"""Self-Tuning module in `Self-Tuning for Data-Efficient Deep Learning (self-tuning, ICML 2021)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/Self-Tuning-for-Data-Efficient-Deep-Learning-icml21.pdf>`_.

    Args:
        encoder_q (Classifier): Query encoder.
        encoder_k (Classifier): Key encoder.
        num_classes (int): Number of classes
        K (int): Queue size. Default: 32
        m (float): Momentum coefficient. Default: 0.999
        T (float): Temperature. Default: 0.07

    Inputs:
        - im_q (tensor): input data fed to `encoder_q`
        - im_k (tensor): input data fed to `encoder_k`
        - labels (tensor): classification labels of input data

    Outputs: pgc_logits, pgc_labels, y_q
        - pgc_logits: projector's predictions on both positive and negative samples
        - pgc_labels: contrastive labels
        - y_q: query classifier's predictions

    Shape:
        - im_q, im_k: (minibatch, *) where * means, any number of additional dimensions
        - labels: (minibatch, )
        - y_q: (minibatch, `num_classes`)
        - pgc_logits: (minibatch, 1 + `num_classes` :math:`\times` `K`, `projection_dim`)
        - pgc_labels: (minibatch, 1 + `num_classes` :math:`\times` `K`)
    """

    def __init__(self, encoder_q, encoder_k, num_classes, K=32, m=0.999, T=0.07):
        super(SelfTuning, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.num_classes = num_classes

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue_list", torch.randn(encoder_q.projection_dim, K * self.num_classes))
        self.queue_list = normalize(self.queue_list, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, h, label):
        # gather keys before updating queue
        batch_size = h.shape[0]
        ptr = int(self.queue_ptr[label])
        real_ptr = ptr + label * self.K
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_list[:, real_ptr:real_ptr + batch_size] = h.T

        # move pointer
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[label] = ptr

    def forward(self, im_q, im_k, labels):
        batch_size = im_q.size(0)
        device = im_q.device

        # compute query features
        h_q, y_q = self.encoder_q(im_q)  # queries: h_q (N x projection_dim)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            h_k, _ = self.encoder_k(im_k)  # keys: h_k (N x projection_dim)

        # compute logits
        # positive logits: Nx1
        logits_pos = torch.einsum('nl,nl->n', [h_q, h_k]).unsqueeze(-1)  # Einstein sum is more intuitive

        # cur_queue_list: queue_size * class_num
        cur_queue_list = self.queue_list.clone().detach()

        logits_neg_list = torch.Tensor([]).to(device)
        logits_pos_list = torch.Tensor([]).to(device)

        for i in range(batch_size):
            neg_sample = torch.cat([cur_queue_list[:, 0:labels[i] * self.K],
                                    cur_queue_list[:, (labels[i] + 1) * self.K:]],
                                   dim=1)
            pos_sample = cur_queue_list[:, labels[i] * self.K: (labels[i] + 1) * self.K]
            ith_neg = torch.einsum('nl,lk->nk', [h_q[i:i + 1], neg_sample])
            ith_pos = torch.einsum('nl,lk->nk', [h_q[i:i + 1], pos_sample])
            logits_neg_list = torch.cat((logits_neg_list, ith_neg), dim=0)
            logits_pos_list = torch.cat((logits_pos_list, ith_pos), dim=0)
            self._dequeue_and_enqueue(h_k[i:i + 1], labels[i])

        # logits: 1 + queue_size + queue_size * (class_num - 1)
        pgc_logits = torch.cat([logits_pos, logits_pos_list, logits_neg_list], dim=1)
        pgc_logits = nn.LogSoftmax(dim=1)(pgc_logits / self.T)

        pgc_labels = torch.zeros([batch_size, 1 + self.K * self.num_classes]).to(device)
        pgc_labels[:, 0:self.K + 1].fill_(1.0 / (self.K + 1))
        return pgc_logits, pgc_labels, y_q
