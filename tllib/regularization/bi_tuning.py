"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from tllib.modules.classifier import Classifier as ClassifierBase


class Classifier(ClassifierBase):
    """Classifier class for Bi-Tuning.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        projection_dim (int, optional): Dimension of the projector head. Default: 128
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        In the training mode,
            - y: classifier's predictions
            - z: projector's predictions
            - hn: normalized features after `bottleneck` layer and before `head` layer
        In the eval mode,
            - y: classifier's predictions

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - y: (minibatch, `num_classes`)
        - z: (minibatch, `projection_dim`)
        - hn: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, projection_dim=128, finetune=True, pool_layer=None):
        head = nn.Linear(backbone.out_features, num_classes)
        head.weight.data.normal_(0, 0.01)
        head.bias.data.fill_(0.0)
        super(Classifier, self).__init__(backbone, num_classes=num_classes, head=head, finetune=finetune,
                                         pool_layer=pool_layer)
        self.projector = nn.Linear(backbone.out_features, projection_dim)
        self.projection_dim = projection_dim

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        h = self.backbone(x)
        h = self.pool_layer(h)
        h = self.bottleneck(h)
        y = self.head(h)
        z = normalize(self.projector(h), dim=1)
        hn = torch.cat([h, torch.ones(batch_size, 1, dtype=torch.float).to(h.device)], dim=1)
        hn = normalize(hn, dim=1)
        if self.training:
            return y, z, hn
        else:
            return y

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.projector.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
        ]

        return params


class BiTuning(nn.Module):
    """
    Bi-Tuning Module in `Bi-tuning of Pre-trained Representations <https://arxiv.org/abs/2011.06182?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29>`_.

    Args:
        encoder_q (Classifier): Query encoder.
        encoder_k (Classifier): Key encoder.
        num_classes (int): Number of classes
        K (int): Queue size. Default: 40
        m (float): Momentum coefficient. Default: 0.999
        T (float): Temperature. Default: 0.07

    Inputs:
        - im_q (tensor): input data fed to `encoder_q`
        - im_k (tensor): input data fed to `encoder_k`
        - labels (tensor): classification labels of input data

    Outputs: y_q, logits_z, logits_y, labels_c
        - y_q: query classifier's predictions
        - logits_z: projector's predictions on both positive and negative samples
        - logits_y: classifier's predictions on both positive and negative samples
        - labels_c: contrastive labels

    Shape:
        - im_q, im_k: (minibatch, *) where * means, any number of additional dimensions
        - labels: (minibatch, )
        - y_q: (minibatch, `num_classes`)
        - logits_z: (minibatch, 1 + `num_classes` x `K`, `projection_dim`)
        - logits_y: (minibatch, 1 + `num_classes` x `K`, `num_classes`)
        - labels_c: (minibatch, 1 + `num_classes` x `K`)
    """

    def __init__(self, encoder_q: Classifier, encoder_k: Classifier, num_classes, K=40, m=0.999, T=0.07):
        super(BiTuning, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.num_classes = num_classes

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_h", torch.randn(encoder_q.features_dim + 1, num_classes, K))
        self.register_buffer("queue_z", torch.randn(encoder_q.projection_dim, num_classes, K))
        self.queue_h = normalize(self.queue_h, dim=0)
        self.queue_z = normalize(self.queue_z, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, h, z, label):
        batch_size = h.shape[0]
        assert self.K % batch_size == 0  # for simplicity

        ptr = int(self.queue_ptr[label])
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_h[:, label, ptr: ptr + batch_size] = h.T
        self.queue_z[:, label, ptr: ptr + batch_size] = z.T

        # move pointer
        self.queue_ptr[label] = (ptr + batch_size) % self.K

    def forward(self, im_q, im_k, labels):
        batch_size = im_q.size(0)
        device = im_q.device
        # compute query features
        y_q, z_q, h_q = self.encoder_q(im_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            y_k, z_k, h_k = self.encoder_k(im_k)

        # compute logits for projection z
        # current positive logits: Nx1
        logits_z_cur = torch.einsum('nc,nc->n', [z_q, z_k]).unsqueeze(-1)
        queue_z = self.queue_z.clone().detach().to(device)
        # positive logits: N x K
        logits_z_pos = torch.Tensor([]).to(device)
        # negative logits: N x ((C-1) x K)
        logits_z_neg = torch.Tensor([]).to(device)

        for i in range(batch_size):
            c = labels[i]
            pos_samples = queue_z[:, c, :]  # D x K
            neg_samples = torch.cat([queue_z[:, 0: c, :], queue_z[:, c + 1:, :]], dim=1).flatten(
                start_dim=1)  # D x ((C-1)xK)
            ith_pos = torch.einsum('nc,ck->nk', [z_q[i: i + 1], pos_samples])  # 1 x D
            ith_neg = torch.einsum('nc,ck->nk', [z_q[i: i + 1], neg_samples])  # 1 x ((C-1)xK)
            logits_z_pos = torch.cat((logits_z_pos, ith_pos), dim=0)
            logits_z_neg = torch.cat((logits_z_neg, ith_neg), dim=0)

            self._dequeue_and_enqueue(h_k[i:i + 1], z_k[i:i + 1], labels[i])

        logits_z = torch.cat([logits_z_cur, logits_z_pos, logits_z_neg], dim=1)  # Nx(1+C*K)

        # apply temperature
        logits_z /= self.T
        logits_z = nn.LogSoftmax(dim=1)(logits_z)

        # compute logits for classification y
        w = torch.cat([self.encoder_q.head.weight.data, self.encoder_q.head.bias.data.unsqueeze(-1)], dim=1)
        w = normalize(w, dim=1)  # C x F

        # current positive logits: Nx1
        logits_y_cur = torch.einsum('nk,kc->nc', [h_q, w.T])  # N x C
        queue_y = self.queue_h.clone().detach().to(device).flatten(start_dim=1).T  # (C * K) x F
        logits_y_queue = torch.einsum('nk,kc->nc', [queue_y, w.T]).reshape(self.num_classes, -1,
                                                                           self.num_classes)  # C x K x C

        logits_y = torch.Tensor([]).to(device)

        for i in range(batch_size):
            c = labels[i]
            # calculate the ith sample in the batch
            cur_sample = logits_y_cur[i:i + 1, c]  # 1
            pos_samples = logits_y_queue[c, :, c]  # K
            neg_samples = torch.cat([logits_y_queue[0: c, :, c], logits_y_queue[c + 1:, :, c]], dim=0).view(
                -1)  # (C-1)*K

            ith = torch.cat([cur_sample, pos_samples, neg_samples])  # 1+C*K
            logits_y = torch.cat([logits_y, ith.unsqueeze(dim=0)], dim=0)

        logits_y /= self.T
        logits_y = nn.LogSoftmax(dim=1)(logits_y)

        # contrastive labels
        labels_c = torch.zeros([batch_size, self.K * self.num_classes + 1]).to(device)
        labels_c[:, 0:self.K + 1].fill_(1.0 / (self.K + 1))
        return y_q, logits_z, logits_y, labels_c
