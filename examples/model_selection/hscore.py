"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""

import os
import sys
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.append('../..')
from tllib.ranking import h_score

sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    logger = utils.Logger(args.data, args.arch, 'results_hscore')
    print(args)
    print(f'Calc Transferabilities of {args.arch} on {args.data}')

    try:
        features = np.load(os.path.join(logger.get_save_dir(), 'features.npy'))
        predictions = np.load(os.path.join(logger.get_save_dir(), 'preds.npy'))
        targets = np.load(os.path.join(logger.get_save_dir(), 'targets.npy'))
        print('Loaded extracted features')
    except:
        print('Conducting feature extraction')
        data_transform = utils.get_transform(resizing=args.resizing)
        print("data_transform: ", data_transform)
        model = utils.get_model(args.arch, args.pretrained).to(device)
        score_dataset, num_classes = utils.get_dataset(args.data, args.root, data_transform, args.sample_rate,
                                                       args.num_samples_per_classes)
        score_loader = DataLoader(score_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  pin_memory=True)
        print(f'Using {len(score_dataset)} samples for ranking')
        features, predictions, targets = utils.forwarding_dataset(score_loader, model,
                                                                  layer=eval(f'model.{args.layer}'), device=device)
        if args.save_features:
            np.save(os.path.join(logger.get_save_dir(), 'features.npy'), features)
            np.save(os.path.join(logger.get_save_dir(), 'preds.npy'), predictions)
            np.save(os.path.join(logger.get_save_dir(), 'targets.npy'), targets)

    print('Conducting transferability calculation')
    result = h_score(features, targets)

    logger.write(
        f'# {result:.4f} # data_{args.data}_sr{args.sample_rate}_sc{args.num_samples_per_classes}_model_{args.arch}_layer_{args.layer}\n')
    print(f'Results saved in {logger.get_result_dir()}')
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking pre-trained models with HScore')

    # dataset
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')
    parser.add_argument('-sc', '--num-samples-per-classes', default=None, type=int,
                        help='number of samples per classes.')
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N', help='mini-batch size (default: 48)')
    parser.add_argument('--resizing', default='res.', type=str)

    # model
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='model to be ranked: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('-l', '--layer', default='fc',
                        help='before which layer features are extracted')
    parser.add_argument('--pretrained', default=None,
                        help="pretrained checkpoint of the backbone. "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument("--save_features", action='store_true',
                        help="whether to save extracted features")

    args = parser.parse_args()
    main(args)
