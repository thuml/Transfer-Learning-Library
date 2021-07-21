from transformers import AutoModel

def get_model(arch, config):
    if arch.startswith("bert"):
        backbone = AutoModel.from_pretrained(arch, config=config, add_pooling_layer=False, from_tf=bool(".ckpt" in arch))
    elif arch.startswith("distilbert"):
        backbone = AutoModel.from_pretrained(arch, config=config, from_tf=bool(".ckpt" in arch))
    else:
        raise NotImplementedError(arch)
    backbone.out_features = config.hidden_size
    return backbone