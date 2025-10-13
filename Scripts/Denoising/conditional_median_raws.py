import os
import shutil
import numpy as np
from scipy.ndimage import median_filter

from SymmetryEstimation.utils import read_raw_image

from bilateral_raw import visualize_images


def conditional_median_raw(raw_img: np.ndarray, radius: int = 3, threshold: float = 50) -> np.ndarray:
    """
    条件中值滤波去除异常像素（死像素/亮点/暗点）

    参数:
        raw_img: 原始 2D raw 图像
        radius: 中值滤波半径，窗口大小 = 2*radius + 1
        threshold: 异常像素判定阈值

    返回:
        去噪后的图像
    """
    # 转 float32 避免 uint16 计算溢出
    raw_img = raw_img.astype(np.float32)

    # 计算中值图像
    median = median_filter(raw_img, size=2 * radius + 1)
    deviation = np.abs(raw_img - median)

    # 替换异常像素
    mask = deviation > threshold
    output = raw_img.copy()
    output[mask] = median[mask]

    print(f"Median: {median.min():.1f}, {median.max():.1f}")
    print(f"Threshold: {threshold}")
    print(f"Pixels removed: {np.sum(mask)}")

    # 返回原数据类型
    return output.astype(raw_img.dtype)


def denoise_raw_file(input_path: str, width: int, height: int, radius: int = 2,
                     threshold: float = 50, dtype=np.uint16, visualize=False) -> np.ndarray:
    """
    读取 raw 并进行条件中值滤波去噪
    """
    img = read_raw_image(input_path, width, height, dtype=dtype)
    denoised = conditional_median_raw(img, radius, threshold)
    denoised = denoised.astype(dtype)
    # 可视化对比（可选）
    if visualize:
        visualize_images(img, denoised)
    return denoised


def process_folder(input_dir: str, output_dir: str, width: int, height: int,
                   radius: int = 2, threshold: float = 50, dtype=np.uint16):
    """
    批量处理文件夹内 raw 文件并保存去噪结果
    """
    # 清空输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 获取所有 raw 文件
    filenames = [f for f in os.listdir(input_dir) if f.endswith(".raw")]
    filenames.sort()

    for i, fname in enumerate(filenames):
        input_path = os.path.join(input_dir, fname)
        print(f"[{i + 1}/{len(filenames)}] Processing {fname} ...")

        denoised = denoise_raw_file(input_path, width, height, radius, threshold, dtype, 0)

        output_path = os.path.join(output_dir, fname)
        denoised.tofile(output_path)
        print(f"Saved denoised: {output_path}")


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0709"
    output_dir = r"D:\Data\cbct\CBCT0709_denoised"
    image_size = 1420

    process_folder(data_dir, output_dir, image_size, image_size,
                   radius=3, threshold=20, dtype=np.uint16)
