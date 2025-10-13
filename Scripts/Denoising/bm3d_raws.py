import os
import time

import bm3d
import numpy as np

from SymmetryEstimation.utils import read_raw_image


def bm3d_denoise_raw(img, sigma=25.0):
    # BM3D 去噪
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

    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    print(f"BM3D Denoising Time: {end_time - start_time:.4f} seconds")

    return denoised


def process_folder(input_dir, output_dir, width, height, sigma=25.0, dtype=np.uint16):
    """
    批量处理文件夹内 raw 文件，并保存去噪结果
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = [f for f in os.listdir(input_dir) if f.endswith(".raw")]
    filenames.sort()  # 按文件名排序处理

    for i, fname in enumerate(filenames):
        input_path = os.path.join(input_dir, fname)
        print(f"[{i + 1}/{len(filenames)}] Processing {fname} ...")

        img = read_raw_image(input_path, width, height, dtype=dtype)
        denoised = bm3d_denoise_raw(img, sigma).astype(dtype)

        output_path = os.path.join(output_dir, fname)
        denoised.tofile(output_path)
        print(f"Saved denoised: {output_path}")


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0709"
    output_dir = r"D:\Data\cbct\CBCT0709_denoised"
    image_size = 1420

    process_folder(data_dir, output_dir, image_size, image_size, sigma=5, dtype=np.uint16)
