import torch
import torch.nn as nn
from dalib.modules.classifier import Classifier as ClassifierBase

__all__ = ['MultipleKernelMaximumMeanDiscrepancy', 'JointMultipleKernelMaximumMeanDiscrepancy', 'ImageClassifier']


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks <https://arxiv.org/pdf/1502.02791>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as

    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},

    where :math:`k` is a kernel function. Using kernel trick, MK-MMD can be computed as

    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}). \\

    For linear computation complexity, we use the unbiased estimate of MK-MMD as follows,

    .. math::
        \hat{D}_{\mathcal{L}}(P, Q) &= \dfrac{2}{n} \sum_{i=1}^{n/2} \left( k(z_{2i-1}^{s}, z_{2i}^{s})
         + k(z_{2i-1}^{t}, z_{2i}^{t}) \right)\\
         &- \dfrac{2}{n} \sum_{i=1}^{n/2} \left( k(z_{2i-1}^{s}, z_{2i}^{t})
         + k(z_{2i-1}^{t}, z_{2i}^{s}) \right),\\

    where :math:`n` is the batch size.

    Parameters:
        - **kernels** (tuple(`nn.Module`)): kernel functions.

    Inputs: z_s, z_t
        - **z_s** (tensor): activations from the source domain, :math:`z^s`
        - **z_t** (tensor): activations from the target domain, :math:`z^t`

    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels.

    Examples::
        >>> from dalib.modules.kernels import GaussianKernel
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """
    def __init__(self, *kernels):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None

    def forward(self, z_s, z_t):
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix).to(z_s.device)
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])   # Add up the matrix of each kernel
        loss = (kernel_matrix * self.index_matrix).sum() / float(batch_size)  # Use quad-tuple for linear complexity
        return loss


class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Joint Multiple Kernel Maximum Mean Discrepancy (JMMD) used in
    `Deep Transfer Learning with Joint Adaptation Networks <https://arxiv.org/abs/1605.06636>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations in layers :math:`\mathcal{L}` as :math:`\{(z_i^{s1}, ..., z_i^{s|\mathcal{L}|})\}_{i=1}^{n_s}` and
    :math:`\{(z_i^{t1}, ..., z_i^{t|\mathcal{L}|})\}_{i=1}^{n_t}`. The empirical estimate of
    :math:`\hat{D}_{\mathcal{L}}(P, Q)` is computed as the squared distance between the empirical kernel mean
    embeddings as

    .. math::
        \hat{D}_{\mathcal{L}}(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{sl}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{tl}, z_j^{tl}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{tl}). \\

    For linear computation complexity, we use the unbiased estimate as follows,

     .. math::
        \hat{D}_{\mathcal{L}}(P, Q) &= \dfrac{2}{n} \sum_{i=1}^{n/2} \left( \prod_{l \in \mathcal{L}} k^l(z_{2i-1}^{sl}, z_{2i}^{sl})
         + \prod_{l \in \mathcal{L}} k^l(z_{2i-1}^{tl}, z_{2i}^{tl}) \right)\\
         &- \dfrac{2}{n} \sum_{i=1}^{n/2} \left( \prod_{l \in \mathcal{L}} k^l(z_{2i-1}^{sl}, z_{2i}^{tl})
         + \prod_{l \in \mathcal{L}} k^l(z_{2i-1}^{tl}, z_{2i}^{sl}) \right),\\

    where :math:`n` is the batch size.

    Parameters:
        - **kernels** (tuple(tuple(`nn.Module`))): kernel functions, where `kernels[r]` corresponds to kernel :math:`k^{\mathcal{L}[r]}`.

    Inputs: z_s, z_t
        - **z_s** (tuple(tensor)): multiple layers' activations from the source domain, :math:`z^s`
        - **z_t** (tuple(tensor)): multiple layers' activations from the target domain, :math:`z^t`

    Shape:
        - :math:`z^{sl}` and :math:`z^{tl}`: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{sl}` and :math:`z^{tl}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels for a certain layer.

    Examples::
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
        >>> layer2_kernels = (GaussianKernel(1.), )
        >>> loss = JointMultipleKernelMaximumMeanDiscrepancy(layer1_kernels, layer2_kernels)
        >>> # layer1 features from source domain and target domain
        >>> z1_s, z1_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # layer2 features from source domain and target domain
        >>> z2_s, z2_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss((z1_s, z2_s), (z1_t, z2_t))
    """
    def __init__(self, *kernels):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None

    def forward(self, z_s, z_t):
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix).to(z_s[0].device)

        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels in zip(z_s, z_t, self.kernels):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            kernel_matrix *= sum([kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        loss = (kernel_matrix * self.index_matrix).sum() / float(batch_size)  # Use quad-tuple for linear complexity
        return loss


def _update_index_matrix(batch_size, index_matrix=None):
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    E.g. when batch_size = 3, index_matrix is
            [[ 0.,  1.,  0.,  0., -1., -1.],
            [ 0.,  0.,  1., -1.,  0., -1.],
            [ 1.,  0.,  0., -1., -1.,  0.],
            [ 0.,  0.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1.],
            [ 0.,  0.,  0.,  1.,  0.,  0.]]
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            index_matrix[s1, s2] = 1.
            index_matrix[t1, t2] = 1.
            index_matrix[s1, t2] = -1.
            index_matrix[s2, t1] = -1.
    return index_matrix


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone, num_classes, bottleneck_dim=256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        head = nn.Linear(bottleneck_dim, num_classes)
        bottleneck[0].weight.data.normal_(0, 0.005)
        bottleneck[0].bias.data.fill_(0.1)
        head.weight.data.normal_(0, 0.01)
        head.bias.data.fill_(0.0)
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, head)


