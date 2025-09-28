import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as nd_shift

from SymmetryEstimation.symmetry_estimation_offset import (
    visualize_sub_matching_scores,
    visualize_matching_scores,
)
from SymmetryEstimation.utils import (
    read_raw_image,
    read_projection_file,
    compute_metrics,
)

# =============================
# 参数设置
# =============================
METRIC = "mse"  # 可选: 'ncc' | 'grad_ncc' | 'ssim' | 'mse'
SHOW_FIG = 0  # 是否显示中间可视化
MAX_SHIFT = 10  # 最大整数像素搜索范围
ROW_STEP = 30  # 取样间隔，用于估算COR
N_REPEAT = 90  # 每行复制次数


def find_best_shift(s1, s2, max_shift, metric=METRIC):
    """整数像素搜索最佳平移"""
    s1, s2 = s1.T, s2.T
    maximize = metric in ("ncc", "grad_ncc", "ssim")

    best_shift, best_score = 0, (-np.inf if maximize else np.inf)
    scores = []

    for offset in range(-max_shift, max_shift + 1):
        s1_shifted = nd_shift(s1, shift=(0, offset), order=1, mode="nearest")
        score = compute_metrics(s1_shifted, s2, metric=metric)

        if (maximize and score > best_score) or (not maximize and score < best_score):
            best_shift, best_score = offset, score
        scores.append(score)

    if SHOW_FIG:
        visualize_matching_scores(scores, best_shift, max_shift)
    return best_shift, best_score


def best_shift_subpixel(s1, s2, search_range=(-2, 2), step=0.1, metric=METRIC):
    """亚像素精细搜索"""
    s1, s2 = s1.T, s2.T
    maximize = metric in ("ncc", "grad_ncc", "ssim")

    best_shift, best_score = 0, (-np.inf if maximize else np.inf)
    scores = []
    offsets = np.arange(search_range[0], search_range[1] + step / 2, step)

    for offset in offsets:
        s1_shifted = nd_shift(s1, shift=(0, offset), order=1, mode="nearest")
        crop = int(np.ceil(abs(offset)))
        # 裁掉边缘避免插值空白
        region1 = s1_shifted[:, crop:-crop] if crop > 0 else s1
        region2 = s2[:, crop:-crop] if crop > 0 else s2
        if region1.shape[1] < 10:  # 区域过小跳过
            continue

        score = compute_metrics(region1, region2, metric=metric)
        if (maximize and score > best_score) or (not maximize and score < best_score):
            best_shift, best_score = offset, score
        scores.append(score)

    if SHOW_FIG:
        visualize_sub_matching_scores(scores, best_shift, offsets)
    return best_shift, best_score


def estimate_rotation_center(sinogram, show=0):
    """
    根据 sinogram 中相隔 180° 的投影对，估算旋转中心
    返回：center_px(像素) , center_shift(亚像素)
    """
    n_angles, width = sinogram.shape
    if n_angles % 2:  # 确保角度数为偶数
        sinogram = sinogram[:-1]
        n_angles -= 1

    diffs, sub_shifts = [], []

    for row in range(1, 180, ROW_STEP):
        # 取相隔 180° 的两行并重复 N 次
        s1 = np.tile(sinogram[row, :], (N_REPEAT, 1)).T
        s2 = np.tile(sinogram[row + 180, :], (N_REPEAT, 1)).T


        from skimage.filters import threshold_otsu

        thresh = threshold_otsu(np.hstack([s1, s2])) if np.max(s2) > 1000 else 10
        s1_bin = (s1 > thresh).astype(np.int8)
        s2_bin = (s2 > thresh).astype(np.int8)

        print(f"阈值: {thresh:.1f}")

        diff = np.abs(s1_bin - s2_bin)
        projection_pair = np.hstack([s1_bin, s2_bin, diff])
        print(np.sum(diff) / N_REPEAT, np.sum(diff[diff > 0]) / N_REPEAT, np.sum(diff[diff < 0]) / N_REPEAT)

        shift, score = find_best_shift(s1_bin, s2_bin, max_shift=MAX_SHIFT, metric=METRIC)
        print(f"最佳平移: {shift} 像素, score: {score:.4f}")

        sub_shift, sub_score = best_shift_subpixel(
            s1_bin, s2_bin,
            search_range=(shift - 1, shift + 1),
            step=0.1,
            metric=METRIC,
        )
        print(f"最佳平移: {sub_shift} 像素, score: {sub_score:.4f}")

        diffs.append(np.sum(diff) / N_REPEAT)
        sub_shifts.append(sub_shift)

        if show:
            plt.figure(figsize=(6, 8))
            plt.subplot(1, 4, 1)
            plt.imshow(s1, cmap='gray', aspect='auto')
            plt.title(f'Row {row} - 0° projection')
            plt.xlabel('Detector column')
            plt.ylabel('Projection index')
            plt.subplot(1, 4, 2)
            plt.imshow(s2, cmap='gray', aspect='auto')
            plt.title(f'Row {row} - 180° projection flipped')
            plt.xlabel('Detector column')
            plt.ylabel('Projection index')
            plt.subplot(1, 4, 3)
            plt.imshow(projection_pair, cmap='gray', aspect='auto')
            plt.title(f'Diff')
            plt.xlabel('Detector column')
            plt.ylabel('Projection index')
            plt.subplot(1, 4, 4)
            plt.imshow(diff, cmap='gray', aspect='auto')
            plt.title(f'Diff abs')
            plt.xlabel('Detector column')
            plt.ylabel('Projection index')
            plt.tight_layout()
            plt.show()

    return np.mean(diffs) / 4, np.mean(sub_shifts) / 2


# =============================
# 主流程
# =============================
if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0703"
    image_size = 1420

    # 1️⃣ 读取投影文件及角度
    file_list, angle_list = read_projection_file(data_dir)
    sort_idx = np.argsort(angle_list)
    files = [file_list[i] for i in sort_idx]
    angles = np.array(angle_list)[sort_idx]
    print("排序后的角度:", angles)

    # 2️⃣ 读取并堆叠投影数据
    projections = []
    for fname in files:
        img = read_raw_image(os.path.join(data_dir, fname), image_size, image_size)
        img = np.clip(img, 1000, 30000)
        projections.append(img)
    projections = np.stack(projections, axis=0)  # (N_angles, H, W)
    print("投影数据形状:", projections.shape)

    # 3️⃣ 构建 sinogram：取中间一行
    mid_row = image_size // 2
    sinogram = projections[:, mid_row, :]  # (N_angles, W)
    print("sinogram 形状:", sinogram.shape)

    # 4️⃣ 可视化
    if SHOW_FIG:
        plt.figure(figsize=(8, 6))
        extent = [0, sinogram.shape[1], angles.min(), angles.max()]
        plt.imshow(sinogram, cmap="gray", aspect="auto",
                   extent=extent, origin="lower")
        plt.xlabel("Detector Pixel")
        plt.ylabel("Projection Angle (°)")
        plt.title("Sinogram (Angle-sorted)")
        plt.colorbar(label="Intensity")
        plt.tight_layout()
        plt.show()

    # 5️⃣ 估算旋转中心
    center_px, center_shift = estimate_rotation_center(sinogram)
    print(f"center_px   : {center_px:.3f} px")
    print(f"center_shift: {center_shift:.3f} px")

    # 6️⃣ 转换为 mm（根据设备比例）
    scale = 900.07 / 1451.42  # 你的设备参数
    detector_shift_mm = center_px * scale * 0.3
    detector_shift_mm2 = center_shift * scale * 0.3
    print(f"探测器偏移(center_px) ≈ {detector_shift_mm:.3f} mm")
    print(f"探测器偏移(center_shift) ≈ {detector_shift_mm2:.3f} mm")
