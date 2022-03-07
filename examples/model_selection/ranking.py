"""
@author: Yong Liu
@contact: liuyong1095556447@163.com
"""

import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('../../..')
from tllib.transferability import LogME, LEEP, NCE, HScore
from tllib.utils.logger import CompleteLogger

sys.path.append('.')
import utils

metric_dict = {
    'LogME': LogME,
    'LEEP': LEEP,
    'NCE': NCE,
    'HScore': HScore
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_pass(score_loader, model, fc_layer):
    """
    A forward forcasting on dataset

    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params fc_layer: the fc layer name of the model, for registering hooks
    
    returns
        features: extracted features of model
        outputs: outputs of model
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
    outputs = torch.cat([x for x in outputs]).numpy()
    targets = torch.cat([x for x in targets]).numpy()
    
    return features, outputs, targets


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    score_transform = utils.get_score_transform(resizing='default')
    print("score_transform: ", score_transform)

    score_dataset, num_classes = utils.get_dataset(args.data, args.root, score_transform, args.sample_rate, args.num_samples_per_classes)
    score_loader = DataLoader(score_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = utils.get_model(args.arch, args.pretrained).to(device)
    
    result = score_model(args, score_loader)
    print(f'{args.metric} of {args.model}: {result}\n')

    logger.close()



def score_model(score_loader, model, args):
    print(f'Calc Transferabilities of {args.model} on {args.dataset}')
    print('Conducting features extraction...')
    features, outputs, targets = forward_pass(score_loader, model, args.layer)
    predictions = F.softmax(outputs)

    print('Conducting transferability calculation...')
    if args.metric in ['LogME', 'HScore']:
        score = metric_dict[args.metric](features, targets)
    elif args.metric== 'LEEP':
        score = metric_dict[args.metric](predictions, targets)
    elif args.metric== 'NCE':
        score = metric_dict[args.metric](np.max(predictions, axis=1)[1], targets)
    else:
        raise NotImplementedError

    return score
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking pre-trained models')
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N', help='mini-batch size (default: 48)')

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

    args = parser.parse_args()
    main(args)