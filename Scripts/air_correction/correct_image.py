import os

import matplotlib.pyplot as plt
import numpy as np

from SymmetryEstimation.utils import read_raw_image


def read_dark_image(filename, image_size):
    image = read_raw_image(filename, image_size, image_size)
    return image


def read_air_image(data_dir, image_size):
    filenames = [f for f in os.listdir(data_dir) if f.endswith(".raw")]
    images = []
    for fname in sorted(filenames):
        filename = os.path.join(data_dir, fname)
        image = read_raw_image(filename, image_size, image_size)
        images.append(image)
    mean_image = np.mean(images, axis=0)
    return mean_image


def visualize_air_correction(dark_image, air_image, cbct_image):
    """显示空气校正可视化"""
    # --- Step 1. 计算基础校正图像 ---
    corrected_air = air_image - dark_image
    corrected_air = np.clip(corrected_air, 1e-3, None)  # 避免分母为 0
    cbct_corrected = (cbct_image - dark_image) / corrected_air
    cbct_corrected = np.clip(cbct_corrected, 0, np.percentile(cbct_corrected, 99))  # 限制高亮异常值

    # --- Step 2. 绘图 ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Air Correction Visualization", fontsize=16, fontweight='bold')

    # (1) Dark
    axes[0, 0].imshow(dark_image, cmap='gray')
    axes[0, 0].set_title("Dark Image")
    axes[0, 0].axis('off')

    # (2) Air Mean
    axes[0, 1].imshow(air_image, cmap='gray')
    axes[0, 1].set_title("Air Image (Mean)")
    axes[0, 1].axis('off')

    # (3) Air - Dark
    im3 = axes[0, 2].imshow(corrected_air, cmap='gray')
    axes[0, 2].set_title("Air - Dark (Corrected)")
    axes[0, 2].axis('off')
    fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # (4) 原始 CBCT
    axes[1, 0].imshow(cbct_image, cmap='gray')
    axes[1, 0].set_title("Raw CBCT Image")
    axes[1, 0].axis('off')

    # (5) 校正后 CBCT
    im5 = axes[1, 1].imshow(cbct_corrected, cmap='gray')
    axes[1, 1].set_title("Air-Corrected CBCT")
    axes[1, 1].axis('off')
    fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im6 = axes[1, 2].imshow(cbct_corrected - cbct_image / np.max(corrected_air), cmap='gray')
    axes[1, 2].set_title("Air-Corrected CBCT - CBCT")
    axes[1, 2].axis('off')
    fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    save_path = "air_correction_preview.png"
    plt.savefig(save_path, dpi=300)

    plt.show()
    print(f"✅ Visualization saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    air_path = r"D:\Data\cbct\air_correction\air_a"
    dark_path = r"D:\Data\cbct\air_correction\dark\dark_a.raw"
    image_path = r"D:\Data\cbct\202510141600"
    image_size = 1420

    # 读取 CBCT 原始图像
    file_list = sorted([f for f in os.listdir(image_path) if f.endswith(".raw")])
    if not file_list:
        raise FileNotFoundError(f"No .raw files found in {image_path}")
    file_path = os.path.join(image_path, file_list[0])
    print(f"→ Using CBCT file: {file_path}")

    # 加载图像
    dark_image = read_dark_image(dark_path, image_size)
    air_image = read_air_image(air_path, image_size)
    cbct_image = read_raw_image(file_path, image_size, image_size)

    # 可视化空气校正
    visualize_air_correction(dark_image, air_image, cbct_image)
