import sys
import time
import timm
from PIL import Image
import os
import os.path as osp
import tqdm

import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.transforms as T

sys.path.append('../../..')
import common.vision.datasets as datasets
import common.vision.models as models
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrained_checkpoint=None):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=True)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    if pretrained_checkpoint:
        print("=> loading pre-trained model from '{}'".format(pretrained_checkpoint))
        pretrained_dict = torch.load(pretrained_checkpoint)
        backbone.load_state_dict(pretrained_dict, strict=False)
    return backbone


def _convert_from_tensorflow_datasets(tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg',
                                                  label_names=('label', )):
    if osp.exists(osp.join(root, list_file)):
        print("Already exists {}. Pass.".format(osp.join(root, list_file)))
    else:
        print("convert from {} to ImageList".format(info.name))
        os.makedirs(osp.join(root, data_dir), exist_ok=True)
        with open(osp.join(root, list_file), "w") as f:
            index = 0
            for ex in tqdm.tqdm(tfds.as_numpy(tensorflow_dataset)):
                image = ex['image']
                if image.shape[2] == 1:
                    image = image.repeat(3, 2)
                im = Image.fromarray(image)
                filename = osp.join(data_dir, "{}.{}".format(index, suffix))
                im.save(osp.join(root, filename))
                f.write(filename)
                for label_name in label_names:
                    f.write(" {}".format(ex[label_name]))
                f.write("\n")
                index += 1


class TensorFlowDataset(datasets.ImageList):
    def __init__(self, tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg', label_name='label', **kwargs):
        _convert_from_tensorflow_datasets(tensorflow_dataset, info, root, data_dir, list_file, suffix, [label_name, ])
        classes = info.features[label_name].names
        super(TensorFlowDataset, self).__init__(root, classes, osp.join(root, list_file), **kwargs)


class SmallnorbAzimuth(TensorFlowDataset):
    def __init__(self, tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg',
                 label_name='label_azimuth', **kwargs):
        super(SmallnorbAzimuth, self).__init__(tensorflow_dataset, info, root, data_dir,
                                               list_file, suffix, label_name, **kwargs)


class SmallnorblElevation(TensorFlowDataset):
    def __init__(self, tensorflow_dataset, info, root, data_dir, list_file, suffix='jpg',
                 label_name='label_elevation', **kwargs):
        super(SmallnorblElevation, self).__init__(tensorflow_dataset, info, root, data_dir,
                                               list_file, suffix, label_name, **kwargs)


# TODO clevr, kitti, dsprites, resisc45, diabetic_retinopathy_detection, eurosat
def get_dataset(dataset_name, root, train_transform, val_transform, sample_rate=100, sample_size=None):
    if dataset_name in datasets.__dict__:
        # load datasets from common.vision.datasets
        dataset = datasets.__dict__[dataset_name]
        train_dataset = dataset(root=root, split='train', sample_rate=sample_rate, download=True, transform=train_transform)
        test_dataset = dataset(root=root, split='test', sample_rate=100, download=True, transform=val_transform)
        num_classes = train_dataset.num_classes
    else:
        # load datasets from tensorflow_datasets
        os.makedirs(root, exist_ok=True)
        os.makedirs(osp.join(root, "imagelist"), exist_ok=True)
        if dataset_name in ['caltech101', 'cifar100', 'dtd', 'oxford_flowers102',
                            'oxford_iiit_pet', 'patch_camelyon', 'sun397', 'svhn_cropped', 'dmlab']:
            data, info = tfds.load(dataset_name, with_info=True)
            dataset = TensorFlowDataset
        elif dataset_name == 'smallnorb_azimuth':
            data, info = tfds.load("smallnorb", with_info=True)
            dataset = SmallnorbAzimuth
        elif dataset_name == 'smallnorb_elevation':
            data, info = tfds.load("smallnorb", with_info=True)
            dataset = SmallnorblElevation
        else:
            raise NotImplementedError(dataset_name)
        train_dataset = dataset(data['train'], info, root, 'train', 'imagelist/train.txt',
                                          transform=train_transform)
        test_dataset = dataset(data['test'], info, root, 'test', 'imagelist/test.txt',
                                         transform=val_transform)
        num_classes = train_dataset.num_classes
        train_dataset = Subset(train_dataset, list(range(sample_size)))
    return train_dataset, test_dataset, num_classes


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


# def gray_to_rgb(x: torch.Tensor):
#     if x.shape[0] == 1:
#         return torch.cat([x, x, x], 0)
#     else:
#         return x


def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False):
    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.2, 1.))
    elif resizing == 'res.':
        transform = T.Resize(224)
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224)
        ])
    elif resizing == "res.sma|crop":
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'inc.crop':
        transform = T.RandomResizedCrop(224)
    elif resizing == 'cif.crop':
        transform = T.Compose([
            T.Resize((224, 224)),
            T.Pad(28),
            T.RandomCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default'):
    if resizing == 'default':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = T.Resize((224, 224))
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

