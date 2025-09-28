import os
import time

import numpy as np
import bm3d
import matplotlib.pyplot as plt

from SymmetryEstimation.utils import read_raw_image


def visualize_bm3d(raw_file, width, height, sigma=25.0, dtype=np.uint16):
    # 1. 读取 raw
    img = read_raw_image(raw_file, width, height, dtype=dtype)

    profile = bm3d.BM3DProfile()
    profile.num_threads = 32

    # 设置其他优化参数
    profile.bs_ht = 8
    profile.step_ht = 6
    profile.max_3d_size_ht = 12  # 增加最大匹配块数以提高精度
    profile.search_window_ht = 20  # 增大搜索窗口，提供更好的匹配效果

    # 您还可以设置其他参数，如噪声强度、去噪方法等：
    profile.filter_strength = 1  # 设置滤波强度（保持默认）
    profile.print_info = False  # 禁用信息打印以提高速度

    # 记录开始时间
    start_time = time.time()

    # 使用传入的 profile 进行 BM3D 去噪
    denoised = bm3d.bm3d(
        img,
        sigma_psd=sigma,
        profile=profile,  # 使用外部配置的 BM3DProfile
        stage_arg=bm3d.BM3DStages.ALL_STAGES
    )
    denoised = denoised.astype(dtype)
    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    print(f"BM3D Denoising Time: {end_time - start_time:.4f} seconds")

    denoised = denoised

    # 4. 作差
    diff = img - denoised
    print(diff.min(), diff.max())

    # 5. 可视化
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("BM3D Denoised")
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Difference (Original - Denoised)")
    plt.imshow(diff, cmap='gray')
    plt.colorbar()
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0709"
    image_size = 1420
    filenames = [f for f in os.listdir(data_dir) if f.endswith(".raw")][::5]

    # 处理第一张 raw 文件并可视化
    visualize_bm3d(os.path.join(data_dir, filenames[0]), image_size, image_size, sigma=25, dtype=np.uint16)
