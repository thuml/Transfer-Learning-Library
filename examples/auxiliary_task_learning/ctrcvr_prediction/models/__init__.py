from .shared_embedding import SharedEmbeddingModel
from .mmoe import MMoEModel


def get_model(name, categorical_field_dims, numerical_num, task_names, embed_dim, expert_num=4):
    if name == 'sharedembedding':
        print("Model: Shared-Embedding")
        return SharedEmbeddingModel(categorical_field_dims, numerical_num, embed_dim=embed_dim,
                                    tower_mlp_dims=(512, 256), task_names=task_names, dropout=0.2)
    elif name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                                tower_mlp_dims=(128, 64), task_names=task_names, expert_num=expert_num, dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)
