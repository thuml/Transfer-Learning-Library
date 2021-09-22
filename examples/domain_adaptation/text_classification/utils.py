import torch
import torch.nn as nn
import time
from common.utils.meter import AverageMeter, ProgressMeter
from common.modules.classifier import Classifier


class Word2VecCNN(nn.Module):
    def __init__(self, pretrained_embeddings, feature_dim):
        super(Word2VecCNN, self).__init__()
        vocab_size, embedding_dim = pretrained_embeddings.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.requires_grad = False

        self.convs = nn.Sequential(
            nn.Conv1d(embedding_dim, feature_dim, 3),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 3),
            nn.ReLU()
        )
        self.out_features = feature_dim

    def forward(self, x):
        x = x.transpose(0, 1)
        vec = self.embedding(x).transpose(1, 2)
        feature = self.convs(vec)
        return feature


class SequenceClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck=None,
                 bottleneck_dim=-1, head=None, finetune=False, pool_layer=None):
        if pool_layer is None:
            pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool1d(output_size=(1,)),
                nn.Flatten()
            )
        if bottleneck is None:
            bottleneck_dim = backbone.out_features
            bottleneck = nn.Sequential(
                nn.Linear(bottleneck_dim, bottleneck_dim),
                nn.ReLU(),
            )
        super(SequenceClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, head, finetune, pool_layer)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0] * 100
    return acc


def validate(model, val_loader, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, accs],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            predictions = model(batch.text)

            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)

            losses.update(loss.item(), batch.text.size(1))
            accs.update(acc.item(), batch.text.size(1))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc {accs.avg:.3f}'.format(accs=accs))

    return accs.avg
