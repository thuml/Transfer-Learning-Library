=======================================
Learning Strategy
=======================================

.. _GroupDRO:

Group Distributionally robust optimization (GroupDRO)
------------------------------------------------------

.. autoclass:: dglib.generalization.groupdro.AutomaticUpdateDomainWeightModule
    :members:

.. _IRM:

Invariant Risk Minimization (IRM)
----------------------------------

.. autoclass:: dglib.generalization.irm.InvariancePenaltyLoss

.. _MLDG:

Meta Learning for Domain Generalization (MLDG)
------------------------------------------------
`Learning to Generalize: Meta-Learning for Domain Generalization (AAAI 2018) <https://arxiv.org/pdf/1710.03463.pdf>`_

Consider there are :math:`S` source domains, at each learning iteration MLDG splits the
original :math:`S` source domains into `meta-train` domains :math:`S_1` and `meta-test` domains :math:`S_2`. The `inner`
objective is cross entropy loss on `meta-train` domains :math:`S_1`. The `outer` (meta-optimization)
objective contains two terms. The first one (which is the same as inner objective) is cross entropy loss on `meta-train`
domains :math:`S_1` with current model parameters :math:`\theta`

.. math::
    \mathbb{E}_{(x,y) \in S_1} l(f(\theta, x), y)
where :math:`l` denotes cross entropy loss, :math:`f(\theta, x)` denotes predictions from model.
The second term is cross entropy loss on `meta-test` domains :math:`S_2` with inner optimized model parameters
:math:`\theta_{updated}`

.. math::
    \mathbb{E}_{(x,y) \in S_2} l(f(\theta_{updated}, x), y)

In this way, MLDG simulates train/test domain shift during training by synthesizing virtual testing domains within
each mini-batch. The outer objective forces that steps to improve training domain performance should
also improve testing domain performance.

.. note::
    Because we need to compute second-order gradient, this full optimization process may take a long time and have
    heavy budget on GPU resource. A first order approximation implementation can be found at
    `DomainBed <https://github.com/facebookresearch/DomainBed>`_.

.. _VREX:

Variance Risk Extrapolation (VREx)
------------------------------------------------
`Out-of-Distribution Generalization via Risk Extrapolation (ICML 2021) <https://arxiv.org/pdf/2003.00688.pdf>`_

VREx shows that reducing differences in risk across training domains can reduce a model’s sensitivity to a wide range
of extreme distributional shifts. Consider there are :math:`S` source domains. At each learning iteration VREx first
computes cross entropy loss on each source domain separately, producing a loss vector :math:`l \in R^S`. The `ERM`
(vanilla cross entropy) loss can be computed as

.. math::
    l_{\text{ERM}} = \frac{1}{S}\sum_{i=1}^S l_i

And the penalty loss is

.. math::
    penalty = \frac{1}{S} \sum_{i=1}^S {(l_i - l_{\text{ERM}})}^2

The final objective is then

.. math::
    objective = l_{\text{ERM}} + \beta * penalty

where :math:`\beta` is the trade off hyper parameter.
