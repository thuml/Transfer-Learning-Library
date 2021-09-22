import time
import warnings
import argparse
import sys
import shutil
from itertools import chain
import os.path as osp

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchtext.legacy import data

sys.path.append('../../..')
from common.utils.logger import CompleteLogger
from common.utils.meter import AverageMeter, ProgressMeter
from dalib.adaptation.mdd import ClassificationMarginDisparityDiscrepancy, GeneralModule
from dalib.modules.grl import WarmStartGradientReverseLayer
from common.utils.analysis import collect_feature, tsne, a_distance

sys.path.append('.')
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SequenceClassifier(GeneralModule):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024,
                 width=1024, grl=None, finetune=False, pool_layer=None):
        grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if grl is None else grl
        if pool_layer is None:
            pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool1d(output_size=(1,)),
                nn.Flatten()
            )
        bottleneck = nn.Sequential(
            pool_layer,
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[1].weight.data.normal_(0, 0.005)
        bottleneck[1].bias.data.fill_(0.1)

        # The classifier head used for final predictions.
        head = nn.Sequential(
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )
        # The adversarial classifier head
        adv_head = nn.Sequential(
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )
        for dep in range(2):
            head[dep * 3].weight.data.normal_(0, 0.01)
            head[dep * 3].bias.data.fill_(0.0)
            adv_head[dep * 3].weight.data.normal_(0, 0.01)
            adv_head[dep * 3].bias.data.fill_(0.0)
        super(SequenceClassifier, self).__init__(backbone, num_classes, bottleneck,
                                              head, adv_head, grl_layer, finetune)

    def get_parameters(self, base_lr=1.0):
        """
        Return a parameters list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer.
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else base_lr},
            {"params": self.bottleneck.parameters(), "lr": 0.1 * base_lr if self.finetune else base_lr},
            {"params": self.head.parameters(), "lr": base_lr},
            {"params": self.adv_head.parameters(), "lr": base_lr}
        ]
        return params


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    TEXT = data.Field(tokenize='spacy',
                      tokenizer_language='en_core_web_sm', fix_length=args.max_length)
    LABEL = data.LabelField()

    def filter(text):
        return text.label in args.labels

    source_data, target_data = data.TabularDataset.splits(
        path=args.root, train=args.source, validation=args.target, format='csv', skip_header=True,
        fields=[('label', LABEL), ('text', TEXT)], filter_pred=filter)

    TEXT.build_vocab(source_data, target_data, vectors=args.embedding, unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(source_data)
    print("embedding shape:", TEXT.vocab.vectors.shape)
    print("label space", LABEL.vocab.stoi)

    train_source_iter = data.BucketIterator(source_data, batch_size=args.batch_size, sort_key=lambda x: len(x.text), shuffle=True, device=device, repeat=True)
    train_target_iter = data.BucketIterator(target_data, batch_size=args.batch_size,
                                              sort_key=lambda x: len(x.text), shuffle=True, device=device, repeat=True)
    val_loader = data.Iterator(target_data, batch_size=args.batch_size, train=False, sort=False, device=device)

    # create model
    pretrain_embedding = TEXT.vocab.vectors
    EMBEDDING_DIM = pretrain_embedding.shape[1]
    # zero the initial weights of the unknown and padding tokens.
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    pretrain_embedding[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    pretrain_embedding[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    backbone = utils.Word2VecCNN(pretrain_embedding, args.feature_dim)
    model = SequenceClassifier(backbone, num_classes=len(LABEL.vocab), bottleneck_dim=args.feature_dim, width=args.feature_dim)
    model = model.to(device)
    print(model)

    # define loss function
    mdd = ClassificationMarginDisparityDiscrepancy(args.margin).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(model.backbone, model.bottleneck).to(device)
        source_feature = collect_feature(train_source_iter, feature_extractor, device, max_num_features=10)
        target_feature = collect_feature(train_target_iter, feature_extractor, device, max_num_features=10)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(model, val_loader, criterion, args)
        print(acc1)
        return

    # define optimizer and lr scheduler
    optimizer = optim.SGD(model.get_parameters(), lr=args.pretrain_lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.pretrain_lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    def pretrain(train_source_iter, epoch):
        batch_time = AverageMeter('Time', ':4.2f')
        data_time = AverageMeter('Data', ':3.1f')
        losses = AverageMeter('Loss', ':3.2f')
        src_accs = AverageMeter('Src Acc', ':3.1f')
        progress = ProgressMeter(
            args.iters_per_epoch,
            [batch_time, data_time, losses, src_accs],
            prefix="Epoch: [{}]".format(epoch))

        model.train()

        end = time.time()
        for i, batch_s in zip(range(args.iters_per_epoch), train_source_iter):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            optimizer.zero_grad()
            y_s, y_s_adv = model(batch_s.text)
            # cls_loss = criterion(y_s, batch_s.label) + criterion(y_s_adv, batch_s.label)
            cls_loss = criterion(y_s, batch_s.label)
            # compute margin disparity discrepancy between domains
            # for adversarial classifier, minimize negative mdd is equal to maximize mdd
            loss = cls_loss

            src_acc = utils.categorical_accuracy(y_s, batch_s.label)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            model.step()

            losses.update(loss.item(), batch_s.text.size(1))
            src_accs.update(src_acc.item(), batch_s.text.size(1))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        return src_accs.avg

    for epoch in range(args.pretrain_epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        pretrain(train_source_iter, epoch)
        # evaluate on validation set
        acc1 = utils.validate(model, val_loader, criterion, args)

    # define optimizer and lr scheduler
    model.finetune = True
    optimizer = optim.SGD(model.get_parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    def train(train_source_iter, train_target_iter, epoch):
        batch_time = AverageMeter('Time', ':4.2f')
        data_time = AverageMeter('Data', ':3.1f')
        losses = AverageMeter('Loss', ':3.2f')
        src_accs = AverageMeter('Src Acc', ':3.1f')
        tgt_accs = AverageMeter('Tgt Acc', ':3.1f')
        progress = ProgressMeter(
            args.iters_per_epoch,
            [batch_time, data_time, losses, src_accs, tgt_accs],
            prefix="Epoch: [{}]".format(epoch))

        model.train()

        end = time.time()
        for i, batch_s, batch_t in zip(range(args.iters_per_epoch), train_source_iter, train_target_iter):
            if batch_s.text.shape[1] != batch_t.text.shape[1]:
                continue
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            optimizer.zero_grad()
            y_s, y_s_adv = model(batch_s.text)
            y_t, y_t_adv = model(batch_t.text)
            cls_loss = criterion(y_s, batch_s.label) + criterion(y_s_adv, batch_s.label) * args.trade_off_adv
            # compute margin disparity discrepancy between domains
            # for adversarial classifier, minimize negative mdd is equal to maximize mdd
            transfer_loss = -mdd(y_s, y_s_adv, y_t, y_t_adv)
            loss = cls_loss + transfer_loss * args.trade_off

            src_acc = utils.categorical_accuracy(y_s, batch_s.label)
            tgt_acc = utils.categorical_accuracy(y_t, batch_t.label)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            model.step()

            losses.update(loss.item(), batch_s.text.size(1))
            src_accs.update(src_acc.item(), batch_s.text.size(1))
            tgt_accs.update(tgt_acc.item(), batch_s.text.size(1))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        return src_accs.avg

    # start training
    best_acc1 = 0
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_source_iter, train_target_iter, epoch)
        # evaluate on validation set
        acc1 = utils.validate(model, val_loader, criterion, args)

        # remember best acc@1 and save checkpoint
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised domain adaptation LDD for text classification task")
    # dataset parameters
    parser.add_argument('--root', default='data/')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--labels', type=str, nargs='+', help='the label names shared between domains')
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )

    # model parameters
    parser.add_argument('--embedding', type=str,
        help="Path to pretrained word embedding",
        default="glove.840B.300d",
    )
    parser.add_argument('--feature-dim', type=int, default=512)
    parser.add_argument('--trade-off', type=float, default=1,
                        help='trade off for the transfer loss')
    parser.add_argument('--trade-off-adv', type=float, default=1.,
                        help='trade off for the source loss of the adv head')
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    # training parameters
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=0.1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.04,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument('--lr-gamma', default=0.0002, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument('--pretrain-epochs', default=5, type=int, metavar='N',
                        help='number of pretrain epochs to run')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=400, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()

    main(args)