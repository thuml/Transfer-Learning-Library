.. _VREX:

Variance Risk Extrapolation (VREx)
------------------------------------------------
`Out-of-Distribution Generalization via Risk Extrapolation (ICML 2021) <https://arxiv.org/pdf/2003.00688.pdf>`_

VREx shows that reducing differences in risk across training domains can reduce a model’s sensitivity to a wide range
of extreme distributional shifts. Consider there are :math:`S` source domains. At each learning iteration VREx first
computes cross entropy loss on each source domain separately, producing a loss vector :math:`l \in R^S`. The `ERM`
(vanilla cross entropy) loss can be computed as

.. math::
    l_{ERM} = \frac{1}{S}\sum_{i=1}^S l_i

And the penalty loss is

.. math::
    penalty = \frac{1}{S} \sum_{i=1}^S {(l_i - l_{ERM})}^2

The final objective is then

.. math::
    objective = l_{ERM} + \beta * penalty

where :math:`\beta` is the trade off hyper parameter.
