"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import os
import argparse
import sys

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    MetadataCatalog
)
from detectron2.data import detection_utils
from detectron2.engine import default_setup, launch
from detectron2.utils.visualizer import ColorMode

sys.path.append('../../..')
import tllib.vision.models.object_detection.meta_arch as models

import utils


def visualize(cfg, args, model):
    for dataset_name in args.test:
        data_loader = build_detection_test_loader(cfg, dataset_name)

        # create folder
        dirname = os.path.join(args.save_path, dataset_name)
        os.makedirs(dirname, exist_ok=True)

        metadata = MetadataCatalog.get(dataset_name)
        n_current = 0

        # switch to eval mode
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if n_current >= args.n_visualizations:
                    break
                batch_predictions = model(batch)
                for per_image, predictions in zip(batch, batch_predictions):
                    instances = predictions["instances"].to(torch.device("cpu"))
                    # only visualize boxes with highest confidence
                    instances = instances[0: args.n_bboxes]
                    # only visualize boxes with confidence exceeding the threshold
                    instances = instances[instances.scores > args.threshold]
                    # visualize in reverse order of confidence
                    index = [i for i in range(len(instances))]
                    index.reverse()
                    instances = instances[index]
                    img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                    img = detection_utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

                    # scale pred_box to original resolution
                    ori_height, ori_width, _ = img.shape
                    height, width = instances.image_size
                    ratio = ori_width / width
                    for i in range(len(instances.pred_boxes)):
                        instances.pred_boxes[i].scale(ratio, ratio)

                    # save original image
                    visualizer = utils.VisualizerWithoutAreaSorting(img, metadata=metadata,
                                                                    instance_mode=ColorMode.IMAGE)
                    output = visualizer.draw_instance_predictions(predictions=instances)

                    filepath = str(n_current) + ".png"
                    filepath = os.path.join(dirname, filepath)
                    output.save(filepath)

                    n_current += 1
                    if n_current >= args.n_visualizations:
                        break


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = models.__dict__[meta_arch](cfg, finetune=True)
    model.to(torch.device(cfg.MODEL.DEVICE))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    visualize(cfg, args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

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
    parser.add_argument('--test', nargs='+', help='test domain(s)')
    parser.add_argument('--save-path', type=str,
                        help='where to save visualization results ')
    parser.add_argument('--n-visualizations', default=100, type=int,
                        help='maximum number of images to visualize (default: 100)')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='confidence threshold of bounding boxes to visualize (default: 0.5)')
    parser.add_argument('--n-bboxes', default=10, type=int,
                        help='maximum number of bounding boxes to visualize in a single image (default: 10)')
    args = parser.parse_args()
    print("Command Line Args:", args)
    args.test = utils.build_dataset(args.test[::2], args.test[1::2])
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
