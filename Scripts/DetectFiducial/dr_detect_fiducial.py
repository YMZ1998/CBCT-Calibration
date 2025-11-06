import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Scripts.DetectFiducial.correct_dr import correct_image
from SymmetryEstimation.utils import read_raw_image


def dr_detect_fiducial(image):
    min_val, max_val = np.min(image), np.max(image)
    norm = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(norm, (5, 5), 3)
    # blurred=cv2.bilateralFilter(blurred, 3, 75, 75)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=15, maxRadius=25
    )

    output = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), -1)

    return circles, output, cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    # data_dir = r"D:\Data\cbct\DR0707\head\1"
    data_dir = r"D:\Data\cbct\DR0707\body\1"
    image_size = 2130
    filenames = ["A.raw", "B.raw"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    plt.suptitle("Fiducial Detection (A/B Images)", fontsize=14)

    for idx, fname in enumerate(sorted(filenames)):
        file_path = os.path.join(data_dir, fname)
        print(f"\n🟡 正在处理第 {idx + 1} 张图像: {file_path}")

        image = read_raw_image(file_path, image_size, image_size)

        image = correct_image(image)
        # image = np.clip(image, 20, 400)
        circles, output, _ = dr_detect_fiducial(image)

        if circles is not None:
            print(f"✅ 检测到 {len(circles[0])} 个圆点：")
            for i, (x, y, r) in enumerate(circles[0]):
                print(f"    点 {i + 1}: x={x}, y={y}, r={r}")
        else:
            print("⚠️ 未检测到圆")

        axes[idx].imshow(output)
        axes[idx].set_title(f"{fname}")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
