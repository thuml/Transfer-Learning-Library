# reference https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py
import argparse
import random
import sys
import shutil
import time

import torch
from datasets import load_metric
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

sys.path.append('../../..')
from common.modules.classifier import SequenceClassifier
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.data import ForeverDataIterator, send_to_device
from common.utils.logger import CompleteLogger
sys.path.append('.')
import get_dataset
import get_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.arch, use_fast=not args.use_slow_tokenizer)
    def transform(text, pair=None):
        tokens = tokenizer(
            text,
            text_pair=pair,
            padding="max_length" if args.pad_to_max_length else False,
            truncation=True,
            max_length=args.max_length)
        return tokens
    # create dataset
    train_source_dataset, train_target_dataset, val_dataset, num_classes = get_dataset.get_dataset(args.data, args.source, args.target, transform, args.root)

    config = AutoConfig.from_pretrained(args.arch, num_labels=num_classes, finetuning_task=args.source)
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = get_models.get_model(args.arch, config)
    model = SequenceClassifier(backbone, num_classes=num_classes, finetune=args.finetune).to(device)
    print(model)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_source_dataset)), 3):
        print(f"Sample {index} of the training set: {train_source_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    train_source_loader = DataLoader(train_source_dataset, shuffle=True, collate_fn=data_collator,
                                     batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=int(args.batch_size/2))
    train_iter = ForeverDataIterator(train_source_loader, device)

    # define optimizer and lr scheduler
    optimizer = AdamW(model.get_parameters(args.lr), weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.epochs * args.iters_per_epoch,
    )

    if args.phase == 'test':
        # resume from the best checkpoint
        if args.load_from is None:
            args.load_from = logger.get_checkpoint_path('best')
        print("Loading model from", args.load_from)
        checkpoint = torch.load(args.load_from, map_location='cpu')
        model.load_state_dict(checkpoint)
        validate(val_loader, model, args)
        return

    # Train!
    print("***** Running training *****")
    print(f"  Num source examples = {len(train_source_dataset)}")
    print(f"  Num target val examples = {len(val_dataset)}")
    print(f"  Instantaneous batch size = {args.batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    best_acc = 0.
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_iter, model, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc = validate(val_loader, model, args)

        # remember best acc@1 and save checkpoint
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
        if acc > best_acc:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc = max(acc, best_acc)
    print("best acc:", best_acc)
    logger.close()


def train(train_iter, model, optimizer, lr_scheduler, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    losses = AverageMeter('Loss', ':3.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        batch = next(train_iter)
        labels = batch.pop("labels")

        # compute output
        y, f = model(**batch)
        loss = F.cross_entropy(y, labels)
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        # compute gradient and do SGD step
        if i % args.gradient_accumulation_steps == 0 or i == len(args.iters_per_epoch) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        losses.update(loss.item(), labels.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model, args: argparse.Namespace):
    # Metrics
    metric_acc = load_metric("accuracy")

    # evaluate on validation set
    model.eval()
    for step, batch in enumerate(val_loader):
        labels = batch.pop("labels")
        batch = send_to_device(batch, device)
        y, _ = model(**batch)
        predictions = y.argmax(dim=-1)
        metric_acc.add_batch(predictions=predictions, references=labels)

    acc = metric_acc.compute()['accuracy']
    print("acc: {}".format(acc))
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    # dataset parameters
    parser.add_argument('--root', default=None)
    parser.add_argument('-d', '--data', metavar='DATA', default='GLUE')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    # model parameters
    parser.add_argument('-a', '--arch', type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="bert-base-cased",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument('--finetune', action='store_true')

    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--load_from', type=str, default=None, help="Where to load checkpoints from during test")
    args = parser.parse_args()

    main(args)