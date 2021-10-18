"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import numpy as np
import cv2


def generate_target(joints, joints_vis, heatmap_size, sigma, image_size):
    """Generate heatamap for joints.

    Args:
        joints: (K, 2)
        joints_vis: (K, 1)
        heatmap_size: W, H
        sigma:
        image_size:

    Returns:

    """
    num_joints = joints.shape[0]
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3
    image_size = np.array(image_size)
    heatmap_size = np.array(heatmap_size)

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if mu_x >= heatmap_size[0] or mu_y >= heatmap_size[1] \
                or mu_x < 0 or mu_y < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight


def keypoint2d_to_3d(keypoint2d: np.ndarray, intrinsic_matrix: np.ndarray, Zc: np.ndarray):
    """Convert 2D keypoints to 3D keypoints"""
    uv1 = np.concatenate([np.copy(keypoint2d), np.ones((keypoint2d.shape[0], 1))], axis=1).T * Zc  # 3 x NUM_KEYPOINTS
    xyz = np.matmul(np.linalg.inv(intrinsic_matrix), uv1).T  # NUM_KEYPOINTS x 3
    return xyz


def keypoint3d_to_2d(keypoint3d: np.ndarray, intrinsic_matrix: np.ndarray):
    """Convert 3D keypoints to 2D keypoints"""
    keypoint2d = np.matmul(intrinsic_matrix, keypoint3d.T).T  # NUM_KEYPOINTS x 3
    keypoint2d = keypoint2d[:, :2] / keypoint2d[:, 2:3]  # NUM_KEYPOINTS x 2
    return keypoint2d


def scale_box(box, image_width, image_height, scale):
    """
    Change `box` to a square box.
    The side with of the square box will be `scale` * max(w, h)
    where w and h is the width and height of the origin box
    """
    left, upper, right, lower = box
    center_x, center_y = (left + right) / 2, (upper + lower) / 2
    w, h = right - left, lower - upper
    side_with = min(round(scale * max(w, h)), min(image_width, image_height))
    left = round(center_x - side_with / 2)
    right = left + side_with - 1
    upper = round(center_y - side_with / 2)
    lower = upper + side_with - 1
    if left < 0:
        left = 0
        right = side_with - 1
    if right >= image_width:
        right = image_width - 1
        left = image_width - side_with
    if upper < 0:
        upper = 0
        lower = side_with -1
    if lower >= image_height:
        lower = image_height - 1
        upper = image_height - side_with
    return left, upper, right, lower


def get_bounding_box(keypoint2d: np.array):
    """Get the bounding box for keypoints"""
    left = np.min(keypoint2d[:, 0])
    right = np.max(keypoint2d[:, 0])
    upper = np.min(keypoint2d[:, 1])
    lower = np.max(keypoint2d[:, 1])
    return left, upper, right, lower


def visualize_heatmap(image, heatmaps, filename):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
    H, W = heatmaps.shape[1], heatmaps.shape[2]
    resized_image = cv2.resize(image, (int(W), int(H)))
    heatmaps = heatmaps.mul(255).clamp(0, 255).byte().cpu().numpy()
    for k in range(heatmaps.shape[0]):
        heatmap = heatmaps[k]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        masked_image = colored_heatmap * 0.7 + resized_image * 0.3
        cv2.imwrite(filename.format(k), masked_image)
        

def area(left, upper, right, lower):
    return max(right - left + 1, 0) * max(lower - upper + 1, 0)


def intersection(box_a, box_b):
    left_a, upper_a, right_a, lower_a = box_a
    left_b, upper_b, right_b, lower_b = box_b
    return max(left_a, left_b), max(upper_a, upper_b), min(right_a, right_b), min(lower_a, lower_b)
