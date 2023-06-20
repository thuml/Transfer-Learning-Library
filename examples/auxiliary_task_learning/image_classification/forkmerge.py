"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import warnings
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from tllib.utils.logger import CompleteLogger
from tllib.utils.data import ForeverDataIterator
from tllib.weighting.forkmerge import ForkMergeWeightedCombiner
from tllib.weighting.task_sampler import *
from tllib.modules.multi_output_module import MultiOutputImageClassifier

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, name, model, save_path, lr, momentum, weight_decay, epochs, task_to_unweighted_probs):
        """
        A wrapper class for an SGD trainer.

        Args:
            name (str): The name of the trainer.
            model: The machine learning model to be trained.
            save_path (str): The file path to save the trained model.
            lr (float): The learning rate for the optimizer.
            momentum (float): The momentum factor for the optimizer.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer.
            epochs (int): The total number of training epochs.
            task_to_unweighted_probs (dict): A dictionary mapping task names to unweighted probabilities.

        """
        self.name = name
        self.model = model
        self.save_path = save_path
        self.epochs = epochs
        self.optimizer = SGD(model.get_parameters(lr), lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.best_acc1 = {}
        self.task_to_unweighted_probs = task_to_unweighted_probs

    def train(self, train_loaders, epoch_start, epoch_end, iters_per_epoch, val_loaders):
        """
        Trains the model using SGD optimization for the specified number of epochs.

        Args:
            train_loaders (dict): A dict of data loaders for the training datasets.
            epoch_start (int): The starting epoch for training.
            epoch_end (int): The ending epoch for training.
            iters_per_epoch (int): The number of iterations per epoch.
            val_loaders (dict): A dict of data loaders for the validation datasets.
        """
        dataset_sampler = SpecifiedProbMultiTaskSampler(train_loaders, 0, self.task_to_unweighted_probs)
        for epoch in range(epoch_start, epoch_end):
            print(self.scheduler.get_lr())
            # train for one epoch
            utils.train(dataset_sampler, self.model, self.optimizer, epoch, iters_per_epoch, args, device)
            self.scheduler.step()

            # evaluate on validation set
            acc1 = utils.validate_all(val_loaders, self.model, args, device)

            # remember best acc@1 and save checkpoint
            if sum(acc1.values()) > sum(self.best_acc1.values()):
                self.save()
                self.best_acc1 = acc1
            print(self.name, "Epoch:", epoch, "lr:", self.scheduler.get_lr()[0], "val_criteria:", round(sum(acc1.values())/len(acc1), 3),
                  "best_val_criteria:", round(sum(self.best_acc1.values())/len(self.best_acc1), 3))

    def test(self, val_loaders):
        """
        Evaluates the trained model on the validation datasets.

        Args:
            val_loaders (dict): A dict of data loaders for the validation datasets.

        Returns:
            tuple: A tuple containing the average accuracy and a dictionary of accuracies for each task.
        """
        acc1 = utils.validate_all(val_loaders, self.model, args, device)
        return sum(acc1.values()) / len(acc1), acc1

    def load_best(self):
        """
        Loads the model state from the saved checkpoint file.
        """
        self.model.load_state_dict(torch.load(self.save_path, map_location='cpu'))

    def save(self):
        """
        Saves the model state to the specified save path.
        """
        torch.save(self.model.state_dict(), self.save_path)

    def __repr__(self):
        return "Trainer (name={}, task_to_unweighted_probs={})".format(self.name, self.task_to_unweighted_probs)


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_all_datasets, train_target_datasets, val_datasets, test_datasets, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.train_tasks, args.test_tasks, train_transform, val_transform)
    train_loaders = {name: ForeverDataIterator(DataLoader(dataset, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.workers, drop_last=True)) for
                    name, dataset in train_all_datasets.items()}
    val_loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                   for name, dataset in val_datasets.items()}
    test_loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                    for name, dataset in test_datasets.items()}

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    heads = nn.ModuleDict({
        dataset_name: nn.Linear(backbone.out_features, num_classes) for dataset_name in args.train_tasks
    })
    model = MultiOutputImageClassifier(backbone, heads, pool_layer=pool_layer, finetune=not args.scratch).to(device)
    if args.pretrained is not None:
        print("Loading from ", args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    print(heads)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)

    if args.phase == 'test':
        acc1 = utils.validate_all(test_loaders, model, args, device)
        print(acc1)
        return

    # create trainers
    # fork model into B branches
    trainers = {}
    target_trainer = trainers["target"] = \
        Trainer("target", deepcopy(model), logger.get_checkpoint_path("target"),
            args.lr, args.momentum, args.wd, args.epochs,
            {name: float(name in args.test_tasks) for name in args.train_tasks}
        )
    trainers["all"] = Trainer("all", deepcopy(model), logger.get_checkpoint_path("all"),
            args.lr, args.momentum, args.wd, args.epochs,
            {name: 1 for name in args.train_tasks}
        )
    if args.fast:
        print("use fast version of ForkMerge")
    else:
        print("use full version of ForkMerge")
        auxiliary_tasks = list(set(args.train_tasks) - set(args.test_tasks))
        if len(auxiliary_tasks) >= 2:
            for aux_task_name in auxiliary_tasks:
                name = "target+{}".format(aux_task_name)
                trainers[name] = Trainer(name, deepcopy(model),
                                  logger.get_checkpoint_path(name),
                                  args.lr, args.momentum, args.wd, args.epochs,
                                  {name: float(name in args.test_tasks + [aux_task_name])
                                   for name in args.train_tasks})
    print(trainers)

    def evaluate_function(theta):
        # evaluate on validation set
        target_trainer.model.load_state_dict(theta)
        val_criteria, _ = target_trainer.test(val_loaders)
        return val_criteria

    combiner = ForkMergeWeightedCombiner(list(trainers.keys()), evaluate_function, args.lambda_space, debug=True)

    epoch_start = 0
    epoch_end = args.epoch_step
    while epoch_start < args.epochs:
        print("Epoch: {}=>{}".format(epoch_start, epoch_end))

        # Dictionary to store the performance of each trainer and the corresponding theta values.
        performance_dict = {}
        theta_dict = {}

        # Iterate over each trainer.
        for name, trainer in trainers.items():
            print("forking branch", name)
            # independent update for each branch
            trainer.train(train_loaders, epoch_start, epoch_end, args.iters_per_epoch, val_loaders)
            # Load the best model state from the saved checkpoint.
            trainer.load_best()
            # Store the model's parameter.
            theta_dict[name] = deepcopy(trainer.model.state_dict())
            # Calculate the performance of the trainer based on the best accuracy achieved.
            performance_dict[name] = sum(trainer.best_acc1.values()) / len(trainer.best_acc1)

        print("merging branches")
        # search the optimal weighting on the validation set and merge parameters
        theta, lambda_dict = combiner.combine(theta_dict, performance_dict)
        print("lambda_dict:", lambda_dict)

        # prune: only keep the branches with the largest weighting values after the first merge
        if len(trainers) > args.pruned_branches:
            print("pruning branch")
            max_values = sorted(lambda_dict.values(), reverse=True)[:args.pruned_branches]
            for key, value in lambda_dict.items():
                if value not in max_values:
                    trainers.pop(key)
            print(trainers)
            combiner = ForkMergeWeightedCombiner(list(trainers.keys()), evaluate_function, args.lambda_space, debug=True)

        # synchronize parameters
        for name, trainer in trainers.items():
            trainer.model.load_state_dict(theta)

        epoch_start = epoch_end
        epoch_end = min(epoch_end + args.epoch_step, args.epochs)

    test_criteria, test_acc1_dict = target_trainer.test(test_loaders)
    print("Test: {} {}".format(test_criteria, test_acc1_dict))

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fork Merge for MultiTask Learning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-tr', '--train_tasks', help='training task(s)', nargs='+')
    parser.add_argument('-ts', '--test_tasks', help='test tasks(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('--sampler', default="ProportionalMultiTaskSampler")
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--examples_cap', type=int, default=None)
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--pretrained', default=None)
    # training parameters
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N',
                        help='mini-batch size (default: 48)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--lambda_space', type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        nargs='+', help="Manually specified subset of the weighting hyper-parameter space")
    parser.add_argument('--epoch_step', type=int, default=5,
                        help="The number of epochs required for each merge (default: 5)")
    parser.add_argument('--pruned_branches', type=int, default=3,
                        help="How many branches to retain after pruning (default: 3)")
    parser.add_argument("--fast", action='store_true',
                        help="Use the fastest version of ForkMerge "
                             "(only for joint optimization, not for task selection)")
    args = parser.parse_args()
    main(args)
