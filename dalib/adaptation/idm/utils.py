"""
Modified from https://github.com/SikaStar/IDM
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch.nn as nn
from .models.dsbn import DSBN1d, DSBN2d, DSBN1d_idm, DSBN2d_idm


def filter_layers(stage):
    layer_names = ['conv', 'layer1', 'layer2', 'layer3', 'layer4', 'bottleneck']
    idm_bn_names = []
    for i in range(len(layer_names)):
        if i >= stage + 1:
            idm_bn_names.append(layer_names[i])
    return idm_bn_names


def convert_dsbn_idm(model, mixup_bn_names, idm=False):
    for _, (child_name, child) in enumerate(model.named_children()):
        # print(child_name)
        idm_flag = idm
        for name in mixup_bn_names:
            if name in child_name:
                idm_flag = True
        if isinstance(child, nn.BatchNorm2d) and not idm_flag:
            m = DSBN2d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, nn.BatchNorm2d) and idm_flag:
            m = DSBN2d_idm(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            m.BN_mix.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, nn.BatchNorm1d) and not idm_flag:
            m = DSBN1d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, nn.BatchNorm1d) and idm_flag:
            m = DSBN1d_idm(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            m.BN_mix.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        else:
            convert_dsbn_idm(child, mixup_bn_names, idm=idm_flag)


def convert_bn_idm(model, use_target=True):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert (not next(model.parameters()).is_cuda)
        if isinstance(child, DSBN2d):
            m = nn.BatchNorm2d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, DSBN2d_idm):
            m = nn.BatchNorm2d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, DSBN1d):
            m = nn.BatchNorm1d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, DSBN1d_idm):
            m = nn.BatchNorm1d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        else:
            convert_bn_idm(child, use_target=use_target)
