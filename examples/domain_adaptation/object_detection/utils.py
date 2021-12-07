"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import numpy as np
import os
import prettytable
from typing import *
from collections import OrderedDict, defaultdict
import tempfile

from detectron2.data import (
    build_detection_test_loader,
)
import detectron2.utils.comm as comm
from detectron2.evaluation import PascalVOCDetectionEvaluator, inference_on_dataset
from detectron2.evaluation.pascal_voc_evaluation import voc_eval
from detectron2.config import get_cfg
from detectron2.engine import default_setup


import common.vision.datasets.object_detection as datasets


class PascalVOCDetectionPerClassEvaluator(PascalVOCDetectionEvaluator):
    """
    Evaluate Pascal VOC style AP with per-class AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        for cls_name, ap in zip(self._class_names, aps[50]):
            ret["bbox"][cls_name] = ap
        return ret


def validate(model, logger, cfg, args):
    results = OrderedDict()
    for dataset_name in args.test:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = PascalVOCDetectionPerClassEvaluator(dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info(results_i)
            table = prettytable.PrettyTable(["class", "AP"])
            for class_name, ap in results_i["bbox"].items():
                table.add_row([class_name, ap])
            logger.info(table.get_string())
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def build_dataset(dataset_categories, dataset_roots):
    """
    Give a sequence of dataset class name and a sequence of dataset root directory,
    return a sequence of built datasets
    """
    dataset_lists = []
    for dataset_category, root in zip(dataset_categories, dataset_roots):
        dataset_lists.append(datasets.__dict__[dataset_category](root).name)
    return dataset_lists


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
