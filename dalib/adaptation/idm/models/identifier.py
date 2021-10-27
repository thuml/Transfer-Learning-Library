"""
Modified from https://github.com/SikaStar/IDM
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from common.vision.models.reid.identifier import ReIdentifier as ReIdentifierBase


class ReIdentifier(ReIdentifierBase):
    r"""Person reIdentifier in `IDM: An Intermediate Domain Module for Domain Adaptive Person Re-ID (ICCV 2021)
    <https://arxiv.org/pdf/2108.02413v1.pdf>`_. During training, model predictions, extracted features as well as
    attention lambda will be returned.
    """
    def __init__(self, *args, **kwargs):
        super(ReIdentifier, self).__init__(*args, **kwargs)

    def forward(self, x, stage=0):
        if self.training:
            f, attention_lam = self.backbone(x, stage)
        else:
            f = self.backbone(x, stage)
        f = self.pool_layer(f)
        bn_f = self.bottleneck(f)
        if not self.training:
            return bn_f
        predictions = self.head(bn_f)
        return predictions, f, attention_lam
