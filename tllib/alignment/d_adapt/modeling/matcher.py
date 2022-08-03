"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
from torch import Tensor, nn

from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple


class MaxOverlapMatcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to one predicted elements.
    """

    def __init__(self):
        pass

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        _, matched_idxs = match_quality_matrix.max(dim=0)

        anchor_labels = match_quality_matrix.new_full(
            (match_quality_matrix.size(1),), -1, dtype=torch.int8
        )

        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        _, pred_inds_with_highest_quality = nonzero_tuple(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        anchor_labels[pred_inds_with_highest_quality] = 1

        return matched_idxs, anchor_labels
