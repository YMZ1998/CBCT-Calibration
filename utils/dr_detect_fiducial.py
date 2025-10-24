import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from SymmetryEstimation.utils import read_raw_image
from utils.correct_dr import correct_image


def dr_detect_fiducial(image):
    """
    在输入图像中检测圆形金标。
    参数:
        image (ndarray): 原始图像（2D数组，short类型）
    返回:
        circles (ndarray): 检测到的圆信息 [x, y, r]
        output (ndarray): 带标记的 BGR 图像
        norm_bgr (ndarray): 归一化的原始灰度图 (BGR)
    """

    # 归一化到 0-255 范围
    min_val, max_val = np.min(image), np.max(image)
    norm = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # 高斯平滑去噪
    blurred = cv2.GaussianBlur(norm, (5, 5), 3)
    # blurred=cv2.bilateralFilter(blurred, 3, 75, 75)

    # 使用霍夫圆检测
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=15, maxRadius=25
    )

    # 可视化结果图
    output = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), -1)

    return circles, output, cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    # === 参数配置 ===
    # data_dir = r"D:\Data\cbct\DR0707\head\1"
    data_dir = r"D:\Data\cbct\DR0707\body\1"
    image_size = 2130
    filenames = ["A.raw", "B.raw"]

    # === 可视化窗口 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    plt.suptitle("Fiducial Detection (A/B Images)", fontsize=14)

    # === 主循环 ===
    for idx, fname in enumerate(sorted(filenames)):
        file_path = os.path.join(data_dir, fname)
        print(f"\n🟡 正在处理第 {idx + 1} 张图像: {file_path}")

        # 读取原始图像
        image = read_raw_image(file_path, image_size, image_size)

        image = correct_image(image)
        # image = np.clip(image, 20, 400)
        # 检测金标圆
        circles, output, _ = dr_detect_fiducial(image)

        # 输出检测结果
        if circles is not None:
            print(f"✅ 检测到 {len(circles[0])} 个圆点：")
            for i, (x, y, r) in enumerate(circles[0]):
                print(f"    点 {i + 1}: x={x}, y={y}, r={r}")
        else:
            print("⚠️ 未检测到圆")

        # 绘制结果
        axes[idx].imshow(output)
        axes[idx].set_title(f"{fname}")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
