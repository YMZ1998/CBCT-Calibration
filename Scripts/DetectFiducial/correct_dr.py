import os

import matplotlib.pyplot as plt
import numpy as np

from SymmetryEstimation.utils import read_raw_image


def correct_image(dr: np.ndarray, max_bright: float = 255, gamma: float = 0.2,
                  alpha_clip_min: float = 0.05, alpha_clip_max: float = 0.5,
                  invert: bool = 0) -> np.ndarray:
    dr = dr.astype(np.int32)  # 避免溢出
    maxval0 = dr.max()
    maxlimit = maxval0 * gamma
    dr_clipped = np.minimum(dr, maxlimit)
    minval = dr_clipped.min()
    maxval = dr_clipped.max()
    print(f"dr minmax: {minval}, {maxval}")
    normalized = (dr_clipped - minval) / (maxval - minval + 1e-8)
    alpha = pow(normalized, 0.5)
    alpha = np.clip(alpha, alpha_clip_min, alpha_clip_max)
    print(f"alpha minmax: {alpha.min()}, {alpha.max()}")
    alpha=alpha/(alpha.max()-alpha.min())
    if invert:
        alpha = 1.0 - alpha
    corrected = max_bright * alpha
    return corrected.astype(np.int16)


if __name__ == "__main__":
    # === 参数 ===
    data_dir = r"D:\Data\cbct\DR0707\body\1"
    filename = "A.raw"
    image_size = 2130

    # === 读取原始图像 ===
    file_path = os.path.join(data_dir, filename)
    file_path2=r"D:\origin.raw"
    image = read_raw_image(file_path2, image_size, image_size)

    # === 归一化 + 亮度校正 ===
    corrected_image = correct_image(image, 255, 0.2, 0.05, 0.8,1)
    # corrected_image = np.clip(corrected_image, 20, 200)

    # === 对比显示 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(corrected_image, cmap='gray')
    axes[1].set_title("Normalized & Corrected")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
