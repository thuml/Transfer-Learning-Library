import torch

def hook_function_wrapper(layer_output_list):
    def hook_function(module, data_input, data_output):
        layer_output_list.append(data_output)
    return hook_function


def register_hook(model, func, hook_layers):
    for name, layer in model.backbone.named_modules():
        if name in hook_layers:
            layer.register_forward_hook(func)

def reg_classifier(model):
    l2_cls = 0.0
    for name, param in model.head.named_parameters():
        l2_cls += 0.5 * torch.norm(param) ** 2
    for name, param in model.bottleneck.named_parameters():
        l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls

def reg_l2sp(model, model_source_weights):
    fea_loss = 0.0
    for name, param in model.backbone.named_parameters():
        fea_loss += 0.5 * torch.norm(param - model_source_weights[name]) ** 2
    return fea_loss

def reg_fea_map(inputs, layer_outputs_source, layer_outputs_target, model_source):
    model_source(inputs)
    fea_loss = 0.0
    for fm_src, fm_tgt in zip(layer_outputs_source, layer_outputs_target):
        b, c, h, w = fm_src.shape
        fea_loss += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
    return fea_loss

def flatten_outputs(fea):
    return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))

def reg_att_fea_map(inputs, layer_outputs_source, layer_outputs_target, model_source, channel_weights):
    model_source(inputs)
    fea_loss = 0.0
    for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source, layer_outputs_target)):
        b, c, h, w = fm_src.shape
        fm_src = flatten_outputs(fm_src)
        fm_tgt = flatten_outputs(fm_tgt)
        div_norm = h * w
        distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
        distance = c * torch.mul(channel_weights[i], distance ** 2) / (h * w)
        fea_loss += 0.5 * torch.sum(distance)
    return fea_loss