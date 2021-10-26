from common.vision.models.reid.identifier import ReIdentifier as ReIdentifierBase


class ReIdentifier(ReIdentifierBase):
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
