import time
import sys
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.append('.')
import dalib.vision.models.segmentation as models
import dalib.vision.datasets.segmentation as datasets
import dalib.vision.datasets.segmentation.transforms as T
from dalib.utils.metric import ConfusionMatrix
from dalib.utils.avgmeter import AverageMeter, ProgressMeter, Meter

plt.rcParams['figure.dpi'] = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    os.makedirs(args.visualize_dir, exist_ok=True)

    image_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    target_dataset = datasets.__dict__[args.target]
    val_target_dataset = target_dataset(
        root=args.target_root, split='val',
        transforms=T.Resize(image_size=args.test_input_size, label_size=args.test_output_size),
        mean=image_mean
    )
    val_target_loader = DataLoader(val_target_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # create model
    model = models.__dict__[args.arch](num_classes=val_target_dataset.num_classes).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            print(checkpoint.keys())
            model.load_state_dict(checkpoint)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).to(device)
    interp_val = nn.Upsample(size=args.test_output_size[::-1], mode='bilinear', align_corners=True)

    confmat = validate(val_target_loader, model, interp_val, criterion, args)
    print(confmat)
    return


def validate(val_loader: DataLoader, model, interp, criterion, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = Meter('Acc', ':3.2f')
    iou = Meter('IoU', ':3.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc, iou],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    confmat = ConfusionMatrix(model.num_classes)

    with torch.no_grad():
        end = time.time()
        img_id = 0
        for i, (x, label) in enumerate(val_loader):
            x = x.to(device)
            label = label.long().to(device)
            # compute output
            output = interp(model(x))
            loss = criterion(output, label)
            # measure accuracy and record loss
            confmat.update(label.flatten(), output.argmax(1).flatten())
            losses.update(loss.item(), x.size(0))

            acc_global, accs, iu = confmat.compute()
            acc.update(accs.mean().item())
            iou.update(iu.mean().item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            if args.visualize_dir is not None:
                images = x.detach().cpu().numpy()
                preds = output.detach().max(dim=1)[1].cpu().numpy()
                targets = label.cpu().numpy()
                for image, pred, target in zip(images, preds, targets):
                    image = val_loader.dataset.decode_input(image)
                    target = val_loader.dataset.decode_target(target)
                    pred = val_loader.dataset.decode_target(pred)

                    image.save(os.path.join(args.visualize_dir, '%d_image.png' % img_id))
                    target.save(os.path.join(args.visualize_dir, '%d_target.png' % img_id))
                    pred.save(os.path.join(args.visualize_dir, '%d_pred.png' % img_id))

                    img_id += 1

    return confmat


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='deeplabv2_resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: deeplabv2_resnet101)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument('--test-input-size', nargs='+', type=int, default=(1024, 512),
                        help='the input image size during test')
    parser.add_argument('--test-output-size', nargs='+', type=int, default=(1024, 512),
                        help='the output image size during test')
    parser.add_argument("--visualize-dir", default=None, type=str,
                        help="Directory to save segmentation results. Not saving if None")
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    args = parser.parse_args()
    print(args)
    main(args)
