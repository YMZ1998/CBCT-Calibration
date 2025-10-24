import numpy as np
import matplotlib.pyplot as plt

from SymmetryEstimation.utils import read_raw_image


def normalize_and_correct_dr_image(dr: np.ndarray, max_bright, k, x0):
    maxlimit = dr.max()*0.4
    dr = np.clip(dr, 0, maxlimit).astype(np.float32)

    minval, maxval = dr.min(), dr.max()
    print(f"原始图像 min/max: {minval}, {maxval}")

    normalized_dr = (dr - minval) / (maxval - minval)

    # sigmoid_dr = 1.0 / (1.0 + np.exp(-k * (normalized_dr - x0)))
    sigmoid_dr = pow(normalized_dr, 0.6)

    sigmoid_dr = 1.0 - sigmoid_dr

    corrected_dr = (max_bright * sigmoid_dr).astype(np.uint16)

    return corrected_dr


def visualize_comparison(original, enhanced, title1="Original", title2="Enhanced"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(title1)
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title(title2)
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = r"D:\Data\cbct\DR0311\45_1\B.raw"
    width, height = 2130, 2130

    image = read_raw_image(filename, width, height, dtype=np.uint16)

    # import SimpleITK as sitk
    # filename2 = r"D:\Data\cbct\DR0301\45\drra.mhd"
    # image = sitk.ReadImage(filename2)
    # image = sitk.GetArrayFromImage(image)
    print("原始图像形状:", image.shape)

    maxlimit = 10000
    minlimit = 0
    max_bright = 5000
    k = 1.2  # 对比度调整
    x0 = 0.2  # 亮度平衡点

    # 处理图像
    enhanced_image = normalize_and_correct_dr_image(image, max_bright, k, x0)

    # 可视化对比
    visualize_comparison(image, enhanced_image, title1="Original RAW Image", title2="Enhanced Image with S-Activation")
