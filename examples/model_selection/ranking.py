"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('../..')
from tllib.transferability import LogME, LEEP, NCE, HScore

sys.path.append('.')
from utils import Logger
import utils



metric_dict = {
    'LogME': LogME,
    'LEEP': LEEP,
    'NCE': NCE,
    'HScore': HScore
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(args):
    logger = Logger(args.data, args.arch, args.metric)
    print(args)

    try:
        features = np.load(os.path.join(logger.get_savedir(), 'features.npy'))
        predictions = np.load(os.path.join(logger.get_savedir(), 'preds.npy'))
        targets = np.load(os.path.join(logger.get_savedir(), 'targets.npy'))
        print('Loaded extracted features')
    except:
        print('Conducting feature extraction')
        score_transform = utils.get_score_transform(resizing='default')
        print("score_transform: ", score_transform)
        model = utils.get_model(args.arch, args.pretrained).to(device)
        score_dataset, num_classes = utils.get_score_dataset(args.data, args.root, score_transform, args.sample_rate, args.num_samples_per_classes)
        score_loader = DataLoader(score_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        print(f'Using {len(score_dataset)} samples for ranking')
        features, predictions, targets = forward_pass(score_loader, model, eval(f'model.{args.layer}'))
        if args.save_features:
            np.save(os.path.join(logger.get_savedir(), 'features.npy'), features)
            np.save(os.path.join(logger.get_savedir(), 'preds.npy'), predictions)
            np.save(os.path.join(logger.get_savedir(), 'targets.npy'), targets)

    result = score_model(features, predictions, targets)
    
    logger.write(f'# {args.arch}:\t{result}\n')
    logger.close()


def forward_pass(score_loader, model, fc_layer):
    """
    A forward forcasting on full dataset

    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params fc_layer: the fc layer name of the model, for registering hooks
    
    returns
        features: extracted features of model
        prediction: probability outputs of model
        targets: ground-truth labels of dataset
    """
    features = []
    outputs = []
    targets = []
    
    def hook_fn_forward(module, input, output):
        features.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())
    
    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)
    
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.cuda()
            _ = model(data)
    
    forward_hook.remove()

    features = torch.cat([x for x in features]).numpy()
    outputs = torch.cat([x for x in outputs])
    predictions = F.softmax(outputs, dim=-1).numpy()
    targets = torch.cat([x for x in targets]).numpy()
    
    return features, predictions, targets


def score_model(features, predictions, targets):
    print(f'Calc Transferabilities of {args.arch} on {args.data}')

    print('Conducting transferability calculation')
    if args.metric in ['LogME', 'HScore']:
        score = metric_dict[args.metric](features, targets)
    elif args.metric== 'LEEP':
        score = metric_dict[args.metric](predictions, targets)
    elif args.metric== 'NCE':
        score = metric_dict[args.metric](np.max(predictions, axis=1), targets)
    else:
        raise NotImplementedError

    return score
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking pre-trained models')

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

    # metrics
    parser.add_argument('-m', '--metric', type=str, default='LogME',
                    choices=['LogME', 'LEEP', 'NCE', 'HScore'],
                    help='metrics to rank the model (default: LogME)')
    
    parser.add_argument("--save_features", action='store_true',
                    help="where to save extracted features")

    args = parser.parse_args()
    main(args)