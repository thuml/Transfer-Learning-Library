"""
`D-adapt: Decoupled Adaptation for Cross-Domain Object Detection <https://openreview.net/pdf?id=VNqaB1g9393>`_.
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import logging
import os
import argparse
import sys
import pprint
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.engine import default_writers, launch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
import detectron2.utils.comm as comm
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    MetadataCatalog
)
from detectron2.utils.events import EventStorage
from detectron2.evaluation import inference_on_dataset

sys.path.append('../../../..')
import tllib.alignment.d_adapt.modeling.meta_arch as models
from tllib.alignment.d_adapt.proposal import ProposalGenerator, ProposalMapper, PersistentProposalList, flatten
from tllib.alignment.d_adapt.feedback import get_detection_dataset_dicts, DatasetMapper

sys.path.append('..')
import utils

import category_adaptation
import bbox_adaptation


def generate_proposals(model, num_classes, dataset_names, cache_root, cfg):
    """Generate foreground proposals and background proposals from `model` and save them to the disk"""
    fg_proposals_list = PersistentProposalList(os.path.join(cache_root, "{}_fg.json".format(dataset_names[0])))
    bg_proposals_list = PersistentProposalList(os.path.join(cache_root, "{}_bg.json".format(dataset_names[0])))
    if not (fg_proposals_list.load() and bg_proposals_list.load()):
        for dataset_name in dataset_names:
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=ProposalMapper(cfg, False))
            generator = ProposalGenerator(num_classes=num_classes)
            fg_proposals_list_data, bg_proposals_list_data = inference_on_dataset(model, data_loader, generator)
            fg_proposals_list.extend(fg_proposals_list_data)
            bg_proposals_list.extend(bg_proposals_list_data)
        fg_proposals_list.flush()
        bg_proposals_list.flush()
    return fg_proposals_list, bg_proposals_list


def generate_category_labels(prop, category_adaptor, cache_filename):
    """Generate category labels for each proposals in `prop` and save them to the disk"""
    prop_w_category = PersistentProposalList(cache_filename)
    if not prop_w_category.load():
        for p in prop:
            prop_w_category.append(p)

        data_loader_test = category_adaptor.prepare_test_data(flatten(prop_w_category))
        predictions = category_adaptor.predict(data_loader_test)
        for p in prop_w_category:
            p.pred_classes = np.array([predictions.popleft() for _ in range(len(p))])
        prop_w_category.flush()
    return prop_w_category


def generate_bounding_box_labels(prop, bbox_adaptor, class_names, cache_filename):
    """Generate bounding box labels for each proposals in `prop` and save them to the disk"""
    prop_w_bbox = PersistentProposalList(cache_filename)
    if not prop_w_bbox.load():
        # remove (predicted) background proposals
        for p in prop:
            keep_indices = (0 <= p.pred_classes) & (p.pred_classes < len(class_names))
            prop_w_bbox.append(p[keep_indices])

        data_loader_test = bbox_adaptor.prepare_test_data(flatten(prop_w_bbox))
        predictions = bbox_adaptor.predict(data_loader_test)
        for p in prop_w_bbox:
            p.pred_boxes = np.array([predictions.popleft() for _ in range(len(p))])
        prop_w_bbox.flush()
    return prop_w_bbox


def train(model, logger, cfg, args, args_cls, args_box):
    model.train()
    distributed = comm.get_world_size() > 1
    if distributed:
        model_without_parallel = model.module
    else:
        model_without_parallel = model

    # define optimizer and lr scheduler
    params = []
    for module, lr in model_without_parallel.get_parameters(cfg.SOLVER.BASE_LR):
        params.extend(
            get_default_optimizer_params(
                module,
                base_lr=lr,
                weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
                bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            )
        )
    optimizer = maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    scheduler = utils.build_lr_scheduler(cfg, optimizer)

    # resume from the last checkpoint
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # generate proposals from detector
    classes = MetadataCatalog.get(args.targets[0]).thing_classes
    cache_proposal_root = os.path.join(cfg.OUTPUT_DIR, "cache", "proposal")
    prop_t_fg, prop_t_bg = generate_proposals(model, len(classes), args.targets, cache_proposal_root, cfg)
    prop_s_fg, prop_s_bg = generate_proposals(model, len(classes), args.sources, cache_proposal_root, cfg)
    model = model.to(torch.device('cpu'))

    # train the category adaptor
    category_adaptor = category_adaptation.CategoryAdaptor(classes, os.path.join(cfg.OUTPUT_DIR, "cls"), args_cls)
    if not category_adaptor.load_checkpoint():
        data_loader_source = category_adaptor.prepare_training_data(prop_s_fg + prop_s_bg, True)
        data_loader_target = category_adaptor.prepare_training_data(prop_t_fg + prop_t_bg, False)
        data_loader_validation = category_adaptor.prepare_validation_data(prop_t_fg + prop_t_bg)
        category_adaptor.fit(data_loader_source, data_loader_target, data_loader_validation)

    # generate category labels for each proposals
    cache_feedback_root = os.path.join(cfg.OUTPUT_DIR, "cache", "feedback")
    prop_t_fg = generate_category_labels(
        prop_t_fg, category_adaptor, os.path.join(cache_feedback_root, "{}_fg.json".format(args.targets[0]))
    )
    prop_t_bg = generate_category_labels(
        prop_t_bg, category_adaptor, os.path.join(cache_feedback_root, "{}_bg.json".format(args.targets[0]))
    )
    category_adaptor.model.to(torch.device("cpu"))

    if args.bbox_refine:
        # train the bbox adaptor
        bbox_adaptor = bbox_adaptation.BoundingBoxAdaptor(classes, os.path.join(cfg.OUTPUT_DIR, "bbox"), args_box)
        if not bbox_adaptor.load_checkpoint():
            data_loader_source = bbox_adaptor.prepare_training_data(prop_s_fg, True)
            data_loader_target = bbox_adaptor.prepare_training_data(prop_t_fg, False)
            data_loader_validation = bbox_adaptor.prepare_validation_data(prop_t_fg)
            bbox_adaptor.validate_baseline(data_loader_validation)
            bbox_adaptor.fit(data_loader_source, data_loader_target, data_loader_validation)

        # generate bounding box labels for each proposals
        cache_feedback_root = os.path.join(cfg.OUTPUT_DIR, "cache", "feedback_bbox")
        prop_t_fg_refined = generate_bounding_box_labels(
            prop_t_fg, bbox_adaptor, classes,
            os.path.join(cache_feedback_root, "{}_fg.json".format(args.targets[0]))
        )
        prop_t_bg_refined = generate_bounding_box_labels(
            prop_t_bg, bbox_adaptor, classes,
            os.path.join(cache_feedback_root, "{}_bg.json".format(args.targets[0]))
        )
        prop_t_fg += prop_t_fg_refined
        prop_t_bg += prop_t_bg_refined
        bbox_adaptor.model.to(torch.device("cpu"))

    if args.reduce_proposals:
        # remove proposals
        prop_t_bg_new = []
        for p in prop_t_bg:
            keep_indices = p.pred_classes == len(classes)
            prop_t_bg_new.append(p[keep_indices])
        prop_t_bg = prop_t_bg_new

        prop_t_fg_new = []
        for p in prop_t_fg:
            prop_t_fg_new.append(p[:20])
        prop_t_fg = prop_t_fg_new

    model = model.to(torch.device(cfg.MODEL.DEVICE))
    # Data loading code
    train_source_dataset = get_detection_dataset_dicts(args.sources)
    train_source_loader = build_detection_train_loader(dataset=train_source_dataset, cfg=cfg)
    train_target_dataset = get_detection_dataset_dicts(args.targets, proposals_list=prop_t_fg+prop_t_bg)

    mapper = DatasetMapper(cfg, precomputed_proposal_topk=1000, augmentations=utils.build_augmentation(cfg, True))
    train_target_loader = build_detection_train_loader(dataset=train_target_dataset, cfg=cfg, mapper=mapper,
                                                       total_batch_size=cfg.SOLVER.IMS_PER_BATCH)

    # training the object detector
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data_s, data_t, iteration in zip(train_source_loader, train_target_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            optimizer.zero_grad()

            # compute losses and gradient on source domain
            loss_dict_s = model(data_s)
            losses_s = sum(loss_dict_s.values())
            assert torch.isfinite(losses_s).all(), loss_dict_s

            loss_dict_reduced_s = {"{}_s".format(k): v.item() for k, v in comm.reduce_dict(loss_dict_s).items()}
            losses_reduced_s = sum(loss for loss in loss_dict_reduced_s.values())
            losses_s.backward()

            # compute losses and gradient on target domain
            loss_dict_t = model(data_t, labeled=False)
            losses_t = sum(loss_dict_t.values())
            assert torch.isfinite(losses_t).all()

            loss_dict_reduced_t = {"{}_t".format(k): v.item() for k, v in comm.reduce_dict(loss_dict_t).items()}
            (losses_t * args.trade_off).backward()

            if comm.is_main_process():
                storage.put_scalars(total_loss_s=losses_reduced_s, **loss_dict_reduced_s, **loss_dict_reduced_t)

            # do SGD step
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # evaluate on validation set
            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                utils.validate(model, logger, cfg, args)
                comm.synchronize()

            if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def main(args, args_cls, args_box):
    logger = logging.getLogger("detectron2")
    cfg = utils.setup(args)

    # dataset
    args.sources = utils.build_dataset(args.sources[::2], args.sources[1::2])
    args.targets = utils.build_dataset(args.targets[::2], args.targets[1::2])
    args.test = utils.build_dataset(args.test[::2], args.test[1::2])

    # create model
    model = models.__dict__[cfg.MODEL.META_ARCHITECTURE](cfg, finetune=args.finetune)
    model.to(torch.device(cfg.MODEL.DEVICE))
    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return utils.validate(model, logger, cfg, args)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    train(model, logger, cfg, args, args_cls, args_box)

    # evaluate on validation set
    return utils.validate(model, logger, cfg, args)


if __name__ == "__main__":
    args_cls, argv = category_adaptation.CategoryAdaptor.get_parser().parse_known_args()
    print("Category Adaptation Args:")
    pprint.pprint(args_cls)

    args_box, argv = bbox_adaptation.BoundingBoxAdaptor.get_parser().parse_known_args(args=argv)
    print("Bounding Box Adaptation Args:")
    pprint.pprint(args_box)

    parser = argparse.ArgumentParser(add_help=True)
    # dataset parameters
    parser.add_argument('-s', '--sources', nargs='+', help='source domain(s)')
    parser.add_argument('-t', '--targets', nargs='+', help='target domain(s)')
    parser.add_argument('--test', nargs='+', help='test domain(s)')
    # model parameters
    parser.add_argument('--finetune', action='store_true',
                        help='whether use 10x smaller learning rate for backbone')
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
             "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument('--trade-off', default=1., type=float,
                        help='trade-off hyper-parameter for losses on target domain')
    parser.add_argument('--bbox-refine', action='store_true',
                        help='whether perform bounding box refinement')
    parser.add_argument('--reduce-proposals', action='store_true',
                        help='whether remove some low-quality proposals.'
                             'Helpful for RetinaNet')
    # training parameters
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0,
                        help="the rank of this machine (unique per machine)")
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args, argv = parser.parse_known_args(argv)
    print("Detection Args:")
    pprint.pprint(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, args_cls, args_box),
    )
