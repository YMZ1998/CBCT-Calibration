import os

import cv2
import numpy as np

from SymmetryEstimation.utils import read_raw_image
from utils.correct_dr import correct_image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def find_darkest_point_with_confidence(image, x, y, radius, sub_radius):
    """在圆形区域内找最暗点并计算置信度"""
    h, w = image.shape
    y1, y2 = max(0, y - radius), min(h, y + radius)
    x1, x2 = max(0, x - radius), min(w, x + radius)

    # 大圆mask
    yy, xx = np.ogrid[y1:y2, x1:x2]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2

    min_val = np.inf
    min_pos = (x, y)

    for yy_c in range(y1, y2):
        for xx_c in range(x1, x2):
            if not mask[yy_c - y1, xx_c - x1]:
                continue
            sy1, sy2 = max(0, yy_c - sub_radius), min(h, yy_c + sub_radius)
            sx1, sx2 = max(0, xx_c - sub_radius), min(w, xx_c + sub_radius)
            sub_region = image[sy1:sy2, sx1:sx2]
            sub_mask = (np.arange(sx1, sx2)[None, :] - xx_c) ** 2 + (
                    np.arange(sy1, sy2)[:, None] - yy_c) ** 2 <= sub_radius ** 2
            avg_val = np.mean(sub_region[sub_mask])
            if avg_val < min_val:
                min_val = avg_val
                min_pos = (xx_c, yy_c)

    # === 计算置信度 ===
    dark_x, dark_y = min_pos
    # 局部区域
    sy1, sy2 = max(0, dark_y - sub_radius), min(h, dark_y + sub_radius)
    sx1, sx2 = max(0, dark_x - sub_radius), min(w, dark_x + sub_radius)
    sub_region = image[sy1:sy2, sx1:sx2]
    sub_mask = (np.arange(sx1, sx2)[None, :] - dark_x) ** 2 + (
            np.arange(sy1, sy2)[:, None] - dark_y) ** 2 <= sub_radius ** 2
    dark_region = sub_region[sub_mask]
    std_local = np.std(dark_region)
    mean_local = np.mean(dark_region)

    # 周围区域亮度（排除暗区）
    neighbor = image[y1:y2, x1:x2][mask]
    mean_neighbor = np.mean(neighbor)

    C = (mean_neighbor - mean_local) / (mean_neighbor + 1e-5)
    S = std_local / (mean_local + 1e-5)
    confidence = sigmoid(5 * C - 2 * S)

    return (dark_x, dark_y), min_val, confidence


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"点击点: ({x}, {y})")

        # -------- 参数设置 --------
        sub_radius = 3  # 小圆模板半径（像素）

        (dark_x, dark_y), min_val, conf = find_darkest_point_with_confidence(image, x, y, radius, sub_radius)
        print(f"最暗点: ({dark_x}, {dark_y}), 灰度: {min_val:.1f}, 置信度: {conf:.3f}")

        # 可视化
        img_show = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.circle(img_show, (x, y), radius, (0, 255, 0), 1)
        cv2.circle(img_show, (dark_x, dark_y), sub_radius, (0, 0, 255), 1)
        cv2.circle(img_show, (dark_x, dark_y), 2, (0, 0, 255), -1)
        cv2.imshow("Image", img_show)


if __name__ == "__main__":
    # ==== 参数设置 ====
    file_path = r"D:\Data\cbct\体模"
    # file_path = r"D:\Data\cbct\CBCT0707"
    file_name = os.listdir(file_path)[10]
    image_size = 1420
    radius = 80  # 搜索半径

    # ==== 读取 raw 图 ====
    image = read_raw_image(os.path.join(file_path, file_name), image_size, image_size)
    image = correct_image(image, 255, 0.2, 0.05, 0.5)
    # image = np.clip(image, 20, 200)
    # image = np.clip(image, 500, 1000)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # ==== 显示图像并等待点击 ====
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", on_mouse)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
