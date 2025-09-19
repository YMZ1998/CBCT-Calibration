import os

import matplotlib.pyplot as plt
import numpy as np

from SymmetryEstimation.utils import read_raw_image, read_projection_file


def estimate_rotation_center(sinogram):
    n_angles, width = sinogram.shape

    if n_angles % 2 != 0:
        sinogram = sinogram[:-1]
        n_angles -= 1

    # 计算每个探测器列的互相关系数
    corr_coeffs = []
    # for col in range(500, width - 500, 30):
    #     s1 = sinogram[:half, col].reshape(-1, 1)  # shape (180,1)
    #     s2 = sinogram[half:, col][::-1].reshape(-1, 1)  # shape (180,1)
    for row in range(180)[::30]:
        s1 = sinogram[row, :].reshape(-1, 1)  # shape (180,1)
        s2 = sinogram[row + 180, :][::-1].reshape(-1, 1)  # shape (180,1)
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

        threshold = 8000
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
        diff[diff <= 1000] = 0
        diff[diff > 1000] = 1
        print(np.sum(diff))
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

        # 归一化互相关系数 (Pearson correlation)
        c = np.corrcoef(s1, s2)[0, 1]
        corr_coeffs.append(c)

    corr_coeffs = np.array(corr_coeffs)
    print(corr_coeffs)
    # 找最大和次大互相关系数及其列索引
    cols = np.arange(500, width - 500, 30)
    # corr_coeffs 对应 cols，而不是 0..len(corr_coeffs)-1
    idx_sorted = np.argsort(corr_coeffs)
    col1 = cols[idx_sorted[-1]]
    col2 = cols[idx_sorted[-2]]
    R1, R2 = corr_coeffs[idx_sorted[-1]], corr_coeffs[idx_sorted[-2]]
    center_x = (R1 * col1 + R2 * col2) / (R1 + R2)
    return center_x


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0703"
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
    # extent 指定纵轴范围为真实角度
    plt.imshow(sinogram, cmap='gray', aspect='auto', extent=extent, origin='lower')
    plt.xlabel('Detector Pixel (列)')
    plt.ylabel('Projection Angle (°)')
    plt.title('Sinogram (Angle-sorted)')
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.show()

    # 如需保存
    # np.save(os.path.join(data_dir, "sinogram.npy"), sinogram)

    # 假设 sinogram 已按真实角度升序排列，形状 (N, W)
    center_px = estimate_rotation_center(sinogram)
    print(f"旋转中心坐标（像素）: {center_px:.3f}")
