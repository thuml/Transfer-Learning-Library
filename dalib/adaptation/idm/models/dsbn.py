import torch
import torch.nn as nn


class DSBN2d(nn.Module):
    def __init__(self, planes):
        super(DSBN2d, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)

    def forward(self, x):
        if not self.training:
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs % 2 == 0)
        split = torch.split(x, int(bs / 2), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out = torch.cat((out1, out2), 0)
        return out


class DSBN1d(nn.Module):
    def __init__(self, planes):
        super(DSBN1d, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm1d(planes)
        self.BN_T = nn.BatchNorm1d(planes)

    def forward(self, x):
        if not self.training:
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs % 2 == 0)
        split = torch.split(x, int(bs / 2), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out = torch.cat((out1, out2), 0)
        return out


class DSBN2d_idm(nn.Module):
    def __init__(self, planes):
        super(DSBN2d_idm, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)
        self.BN_mix = nn.BatchNorm2d(planes)

    def forward(self, x):
        if not self.training:
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs % 3 == 0)
        split = torch.split(x, int(bs / 3), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out3 = self.BN_mix(split[2].contiguous())
        out = torch.cat((out1, out2, out3), 0)
        return out


class DSBN1d_idm(nn.Module):
    def __init__(self, planes):
        super(DSBN1d_idm, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm1d(planes)
        self.BN_T = nn.BatchNorm1d(planes)
        self.BN_mix = nn.BatchNorm1d(planes)

    def forward(self, x):
        if not self.training:
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs % 3 == 0)
        split = torch.split(x, int(bs / 3), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out3 = self.BN_mix(split[2].contiguous())
        out = torch.cat((out1, out2, out3), 0)
        return out
