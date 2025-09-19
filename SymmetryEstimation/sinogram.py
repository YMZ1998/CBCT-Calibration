import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as nd_shift

from SymmetryEstimation.symmetry_estimation_offset import visualize_sub_matching_scores, visualize_matching_scores
from SymmetryEstimation.utils import read_raw_image, read_projection_file, compute_metrics

show = 0
metric = 'ssim'


def find_best_shift(s1, s2, max_shift, metric='ssim'):
    """
    s1, s2 : 1D numpy array (shape (W,) or (W,1))
    """
    s1 = s1.ravel().astype(np.float32)
    s2 = s2.ravel().astype(np.float32)

    maximize = metric == 'ncc' or metric == 'grad_ncc' or metric == 'ssim'
    best_score = -np.inf if maximize else np.inf
    best_shift = 0
    matching_scores = []
    for shift in range(-max_shift, max_shift + 1):
        if shift > 0:
            # s2 向右移：取 s1[shift:] 与 s2[:-shift]
            a = s1[shift:]
            b = s2[:-shift]
        elif shift < 0:
            # s2 向左移：取 s1[:shift] 与 s2[-shift:]
            a = s1[:shift]
            b = s2[-shift:]
        else:
            a = s1
            b = s2

        if a.size == 0:  # 没有重叠就跳过
            continue
        score = compute_metrics(a, b, metric=metric)
        # score = ssim(s1, s2_shifted, data_range=s1.max() - s1.min())
        print(f"{shift}: {score:.4f}")

        if maximize:
            if score > best_score:
                best_score = score
                best_shift = shift
        else:
            if score < best_score:
                best_score = score
                best_shift = shift

        matching_scores.append(score)
    if show:
        visualize_matching_scores(matching_scores, best_shift, max_shift)
    return best_shift, best_score


def best_shift_subpixel(s1, s2, search_range=(-10, 10), step=0.1, metric='ssim'):
    s1 = s1.ravel().astype(np.float32)
    s2 = s2.ravel().astype(np.float32)
    maximize = metric == 'ncc' or metric == 'grad_ncc' or metric == 'ssim'
    best_score = -np.inf if maximize else np.inf
    best_shift = 0
    sub_matching_scores = []
    sub_range = np.arange(search_range[0], search_range[1] + step / 2, step)
    for shift in sub_range:
        # 将整数平移部分分离出来
        int_shift = int(np.floor(shift))
        frac_shift = shift - int_shift

        if shift >= 0:
            # s2 向右平移
            a = s1[int_shift:]  # s1 的重合部分
            b = s2[:len(a)]  # s2 对应部分
        else:
            # s2 向左平移
            a = s1[:len(s1) + int_shift]
            b = s2[-int_shift:len(s2)]

        if a.size == 0:
            continue

        # 在重合段内再做小数平移
        if abs(frac_shift) > 1e-6:
            b = nd_shift(b, shift=frac_shift, order=1, mode='nearest')
        score = compute_metrics(a, b, metric=metric)
        print(f"{shift}: {score:.4f}")
        if maximize:
            if score > best_score:
                best_score = score
                best_shift = shift
        else:
            if score < best_score:
                best_score = score
                best_shift = shift

        sub_matching_scores.append(score)
    if show:
        visualize_sub_matching_scores(sub_matching_scores, best_shift, sub_range)
    return best_shift, best_score


def estimate_rotation_center(sinogram):
    n_angles, width = sinogram.shape

    if n_angles % 2 != 0:
        sinogram = sinogram[:-1]
        n_angles -= 1

    diffs = []
    sub_shifts = []
    for row in range(180)[::60]:
        s1 = sinogram[row, :].reshape(-1, 1)
        s2 = sinogram[row + 180, :][::-1].reshape(-1, 1)

        shift, score = find_best_shift(s1, s2, max_shift=10, metric=metric)
        print(f"最佳平移: {shift} 像素,  score: {score:.4f}")
        sub_shift, sub_score = best_shift_subpixel(s1, s2, search_range=(shift - 1, shift + 1), step=0.1, metric=metric)
        print(f"最佳平移: {sub_shift} 像素,  score: {sub_score:.4f}")
        sub_shifts.append(sub_shift)
        # 合并显示
        # projection_pair = np.hstack([s1, s2, s1 - s2])
        # print(projection_pair.shape)
        # plt.figure(figsize=(4, 8))
        # plt.imshow(projection_pair, cmap='gray', aspect='auto')
        # plt.xlabel('0° vs 180° flipped')
        # plt.ylabel('Projection index')
        # plt.title(f'Column {row} Projection Comparison')
        # plt.show()

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

        threshold = 6000
        s1 = (s1 > threshold).astype(np.int8)
        s2 = (s2 > threshold).astype(np.int8)
        diff = (s1 - s2)
        projection_pair = np.hstack([s1, s2, diff])
        print(np.sum(diff[diff > 0]), np.sum(diff[diff < 0]))
        plt.subplot(1, 4, 3)
        plt.imshow(projection_pair, cmap='gray', aspect='auto')
        plt.title(f'Diff')
        plt.xlabel('Detector column')
        plt.ylabel('Projection index')

        diff = abs(s1 - s2)
        # diff[diff <= 1000] = 0
        # diff[diff > 1000] = 1
        print(np.sum(diff), np.sum(diff) / 4)
        plt.subplot(1, 4, 4)
        plt.imshow(diff, cmap='gray', aspect='auto')
        plt.title(f'Diff abs')
        plt.xlabel('Detector column')
        plt.ylabel('Projection index')

        plt.tight_layout()
        # plt.show()

        # plt.figure(figsize=(8, 4))
        # plt.plot(s1, label='0°-180°前半列 s1')
        # plt.plot(s2, label='180°后半列翻转 s2', linestyle='--')
        # plt.title(f'Column {col} Symmetry Check')
        # plt.xlabel('Projection index')
        # plt.ylabel('Normalized intensity')
        # plt.legend()
        # plt.show()

        diffs.append(np.sum(diff) / 4)

    return np.mean(diffs), np.mean(sub_shifts) / 2


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0703"
    # data_dir = r"D:\Data\cbct\CBCT0331\A"
    image_size = 1420

    # 1️⃣ 读取文件名和对应角度
    proj_file_list, angle_list = read_projection_file(data_dir)

    # 2️⃣ 按角度排序，得到排序索引
    angle_array = np.array(angle_list)

    sort_idx = np.argsort(angle_array)
    sorted_files = [proj_file_list[i] for i in sort_idx]
    sorted_angles = angle_array[sort_idx]
    print("排序后的角度:", sorted_angles)

    # 3️⃣ 读取投影并堆叠
    projections = []
    for fname in sorted_files:
        img = read_raw_image(os.path.join(data_dir, fname), image_size, image_size)
        img = np.clip(img, 1000, 10000)
        projections.append(img)
    projections = np.stack(projections, axis=0)  # (N_angles, H, W)
    print("投影数据形状:", projections.shape)

    # 4️⃣ 构建正弦图（选取中间行）
    row_idx = image_size // 2
    sinogram = projections[:, row_idx, :]  # shape (N_angles, W)
    print("正弦图形状:", sinogram.shape)

    # 5️⃣ 可视化，纵轴使用真实角度
    # plt.figure(figsize=(8, 6))
    # extent = [0, sinogram.shape[1], sorted_angles.min(), sorted_angles.max()]
    # plt.imshow(sinogram, cmap='gray', aspect='auto', extent=extent, origin='lower')
    # plt.xlabel('Detector Pixel (列)')
    # plt.ylabel('Projection Angle (°)')
    # plt.title('Sinogram (Angle-sorted)')
    # plt.colorbar(label='Intensity')
    # plt.tight_layout()
    # plt.show()

    center_px, center_shift = estimate_rotation_center(sinogram)
    print(f"center_px : {center_px:.3f}")
    print(f"center_shift : {center_shift:.3f}")
    scale = 900.07 / 1451.42
    detector_shift_mm = center_px * scale * 0.3
    detector_shift_mm2 = center_shift * scale * 0.3
    print(f"实际探测器偏移 center_px ≈ {detector_shift_mm:.3f} mm")
    print(f"实际探测器偏移 center_shift ≈ {detector_shift_mm2:.3f} mm")
