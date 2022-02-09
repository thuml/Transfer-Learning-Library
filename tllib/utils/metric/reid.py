# TODO: add documentation
"""
Modified from https://github.com/yxgeee/MMT
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import os
import os.path as osp
from collections import defaultdict
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tllib.utils.meter import AverageMeter, ProgressMeter


def unique_sample(ids_dict, num):
    """Randomly choose one instance for each person id, these instances will not be selected again"""
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams, topk=100, separate_camera_set=False,
        single_gallery_shot=False, first_match_break=False):
    """Compute Cumulative Matching Characteristics (CMC)"""
    dist_mat = dist_mat.cpu().numpy()
    m, n = dist_mat.shape
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(dist_mat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams):
    """Compute mean average precision (mAP)"""
    dist_mat = dist_mat.cpu().numpy()
    m, n = dist_mat.shape
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(dist_mat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -dist_mat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """Perform re-ranking with distance matrix between query and gallery images `q_g_dist`, distance matrix between
    query and query images `q_q_dist` and distance matrix between gallery and gallery images `g_g_dist`.
    """
    q_g_dist = q_g_dist.cpu().numpy()
    q_q_dist = q_q_dist.cpu().numpy()
    g_g_dist = g_g_dist.cpu().numpy()

    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def extract_reid_feature(data_loader, model, device, normalize, print_freq=200):
    """Extract feature for person ReID. If `normalize` is True, `cosine` distance will be employed as distance
    metric, otherwise `euclidean` distance.
    """
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time],
        prefix='Collect feature: ')

    # switch to eval mode
    model.eval()
    feature_dict = dict()

    with torch.no_grad():
        end = time.time()
        for i, (images_batch, filenames_batch, _, _) in enumerate(data_loader):

            images_batch = images_batch.to(device)
            features_batch = model(images_batch)
            if normalize:
                features_batch = F.normalize(features_batch)

            for filename, feature in zip(filenames_batch, features_batch):
                feature_dict[filename] = feature

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

    return feature_dict


def pairwise_distance(feature_dict, query, gallery):
    """Compute pairwise distance between two sets of features"""

    # concat features and convert to pytorch tensor
    # we compute pairwise distance metric on cpu because it may require a large amount of GPU memory, if you are using
    # gpu with a larger capacity, it's faster to calculate on gpu
    x = torch.cat([feature_dict[f].unsqueeze(0) for f, _, _ in query], dim=0).cpu()
    y = torch.cat([feature_dict[f].unsqueeze(0) for f, _, _ in gallery], dim=0).cpu()
    m, n = x.size(0), y.size(0)
    # flatten
    x = x.view(m, -1)
    y = y.view(n, -1)
    # compute dist_mat
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t() - \
               2 * torch.matmul(x, y.t())
    return dist_mat


def evaluate_all(dist_mat, query, gallery, cmc_topk=(1, 5, 10), cmc_flag=False):
    """Compute CMC score, mAP and return"""
    query_ids = [pid for _, pid, _ in query]
    gallery_ids = [pid for _, pid, _ in gallery]
    query_cams = [cid for _, _, cid in query]
    gallery_cams = [cid for _, _, cid in gallery]

    # Compute mean AP
    mAP = mean_ap(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if not cmc_flag:
        return mAP

    cmc_configs = {
        'config': dict(separate_camera_set=False, single_gallery_shot=False, first_match_break=True)
    }
    cmc_scores = {name: cmc(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams, **params) for name, params in
                  cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['config'][k - 1]))
    return cmc_scores['config'][0], mAP


def validate(val_loader, model, query, gallery, device, criterion='cosine', cmc_flag=False, rerank=False):
    assert criterion in ['cosine', 'euclidean']
    # when criterion == 'cosine', normalize feature of single image into unit norm
    normalize = (criterion == 'cosine')

    feature_dict = extract_reid_feature(val_loader, model, device, normalize)
    dist_mat = pairwise_distance(feature_dict, query, gallery)
    results = evaluate_all(dist_mat, query=query, gallery=gallery, cmc_flag=cmc_flag)
    if not rerank:
        return results
    # apply person re-ranking
    print('Applying person re-ranking')
    dist_mat_query = pairwise_distance(feature_dict, query, query)
    dist_mat_gallery = pairwise_distance(feature_dict, gallery, gallery)
    dist_mat = re_ranking(dist_mat, dist_mat_query, dist_mat_gallery)
    return evaluate_all(dist_mat, query=query, gallery=gallery, cmc_flag=cmc_flag)


# location parameters for visualization
GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
# border width
BW = 5
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(data_loader, model, query, gallery, device, visualize_dir, criterion='cosine',
                             rerank=False, width=128, height=256, topk=10):
    """Visualize ranker results. We first compute pair-wise distance between query images and gallery images. Then for
    every query image, `topk` gallery images with least distance between given query image are selected. We plot the
    query image and selected gallery images together. A green border denotes a match, and a red one denotes a mis-match.
    """
    assert criterion in ['cosine', 'euclidean']
    normalize = (criterion == 'cosine')

    # compute pairwise distance matrix
    feature_dict = extract_reid_feature(data_loader, model, device, normalize)
    dist_mat = pairwise_distance(feature_dict, query, gallery)

    if rerank:
        dist_mat_query = pairwise_distance(feature_dict, query, query)
        dist_mat_gallery = pairwise_distance(feature_dict, gallery, gallery)
        dist_mat = re_ranking(dist_mat, dist_mat_query, dist_mat_gallery)

    # make dir if not exists
    os.makedirs(visualize_dir, exist_ok=True)

    dist_mat = dist_mat.numpy()
    num_q, num_g = dist_mat.shape
    print('query images: {}'.format(num_q))
    print('gallery images: {}'.format(num_g))

    assert num_q == len(query)
    assert num_g == len(gallery)

    # start visualizing
    import cv2
    sorted_idxes = np.argsort(dist_mat, axis=1)
    for q_idx in range(num_q):
        q_img_path, q_pid, q_cid = query[q_idx]

        q_img = cv2.imread(q_img_path)
        q_img = cv2.resize(q_img, (width, height))
        # use black border to denote query image
        q_img = cv2.copyMakeBorder(
            q_img, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        q_img = cv2.resize(q_img, (width, height))
        num_cols = topk + 1
        grid_img = 255 * np.ones(
            (height, num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING, 3), dtype=np.uint8
        )
        grid_img[:, :width, :] = q_img

        # collect top-k gallery images with smallest distance
        rank_idx = 1
        for g_idx in sorted_idxes[q_idx, :]:
            g_img_path, g_pid, g_cid = gallery[g_idx]
            invalid = (q_pid == g_pid) & (q_cid == g_cid)
            if not invalid:
                matched = (g_pid == q_pid)
                border_color = GREEN if matched else RED
                g_img = cv2.imread(g_img_path)
                g_img = cv2.resize(g_img, (width, height))
                g_img = cv2.copyMakeBorder(
                    g_img, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color
                )
                g_img = cv2.resize(g_img, (width, height))
                start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                end = (rank_idx + 1) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                grid_img[:, start:end, :] = g_img

                rank_idx += 1
                if rank_idx > topk:
                    break

        save_path = osp.basename(osp.splitext(q_img_path)[0])
        cv2.imwrite(osp.join(visualize_dir, save_path + '.jpg'), grid_img)

        if (q_idx + 1) % 100 == 0:
            print('Visualize {}/{}'.format(q_idx + 1, num_q))

    print('Visualization process is done, ranked results are saved to {}'.format(visualize_dir))
