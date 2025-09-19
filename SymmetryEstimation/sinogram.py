import os

import matplotlib.pyplot as plt
import numpy as np

from SymmetryEstimation.utils import read_raw_image, read_projection_file


def estimate_rotation_center(sinogram):
    n_angles, width = sinogram.shape

    if n_angles % 2 != 0:
        sinogram = sinogram[:-1]
        n_angles -= 1

    diffs = []
    for row in range(180)[::30]:
        s1 = sinogram[row, :].reshape(-1, 1)
        s2 = sinogram[row + 180, :][::-1].reshape(-1, 1)
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
        projection_pair = np.hstack([s1, s2, abs(s1 - s2)])
        diff = abs(s1 - s2)
        print(np.sum(diff))
        plt.subplot(1, 4, 3)
        plt.imshow(projection_pair, cmap='gray', aspect='auto')
        plt.title(f'Diff')
        plt.xlabel('Detector column')
        plt.ylabel('Projection index')

        diff = abs(s1 - s2)
        # diff[diff <= 1000] = 0
        # diff[diff > 1000] = 1
        print(np.sum(diff) / 4)
        plt.subplot(1, 4, 4)
        plt.imshow(diff, cmap='gray', aspect='auto')
        plt.title(f'Diff abs')
        plt.xlabel('Detector column')
        plt.ylabel('Projection index')

        plt.tight_layout()
        plt.show()

        # plt.figure(figsize=(8, 4))
        # plt.plot(s1, label='0°-180°前半列 s1')
        # plt.plot(s2, label='180°后半列翻转 s2', linestyle='--')
        # plt.title(f'Column {col} Symmetry Check')
        # plt.xlabel('Projection index')
        # plt.ylabel('Normalized intensity')
        # plt.legend()
        # plt.show()

        diffs.append(np.sum(diff) / 4)

    return np.mean(diffs)


if __name__ == "__main__":
    # data_dir = r"D:\Data\cbct\CBCT0703"
    data_dir = r"D:\Data\cbct\CBCT0331\A"
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
    plt.figure(figsize=(8, 6))
    extent = [0, sinogram.shape[1], sorted_angles.min(), sorted_angles.max()]
    plt.imshow(sinogram, cmap='gray', aspect='auto', extent=extent, origin='lower')
    plt.xlabel('Detector Pixel (列)')
    plt.ylabel('Projection Angle (°)')
    plt.title('Sinogram (Angle-sorted)')
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.show()


    center_px = estimate_rotation_center(sinogram)
    print(f"旋转中心坐标（像素）: {center_px:.3f}")
    scale = 900.07 / 1451.42
    detector_shift_mm = center_px * scale * 0.3
    print(f"实际探测器偏移 ≈ {detector_shift_mm:.3f} mm")
