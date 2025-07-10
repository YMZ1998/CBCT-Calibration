import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from multi_symmetry_estimation_offset import generate_symmetric_angle_pairs
from utils import read_raw_image, read_projection_file

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def detect_circles(image, fliter=1, y_center=784, y_tolerance=10):
    """
    在图像中检测圆形，并仅保留 y 坐标在指定范围内的圆
    """
    image = np.clip(image, 500, 1500)
    norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=15, maxRadius=25)

    output = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    norm = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        if fliter:
            filtered = np.array([c for c in circles if abs(c[1] - y_center) <= y_tolerance])
        else:
            filtered = circles

        for (x, y, r) in filtered:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), -1)

        return np.expand_dims(filtered, axis=0), output, norm
    else:
        return None, output, norm


def select_from_detected_circles(image, circles):
    """
    允许用户点击选择某个圆点
    """
    image = np.clip(image, 500, 1500)
    norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    for i, (x, y, r) in enumerate(circles[0]):
        cv2.circle(image_rgb, (int(x), int(y)), int(r), (0, 255, 0), 1)
        cv2.putText(image_rgb, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 0, 0), 1, cv2.LINE_AA)

    plt.imshow(image_rgb)
    plt.title("点击你想要的圆点编号")
    pts = plt.ginput(1, timeout=0)
    plt.close()

    if pts:
        x_click, y_click = pts[0]
        distances = np.linalg.norm(circles[0][:, :2] - [x_click, y_click], axis=1)
        closest_idx = np.argmin(distances)
        x, y, r = circles[0][closest_idx]
        print(f"你选择的是第 {closest_idx} 个圆点: ({x:.1f}, {y:.1f})")
        return np.array([x, y])
    else:
        print("⚠️ 未选择任何点")
        return None


def match_and_estimate_offset(img1, img2, show_plot=1):
    """
    匹配两个图像中的圆点，估计 x/y 偏移量
    """
    circles1, _, norm1 = detect_circles(img1)
    circles2, _, _ = detect_circles(np.fliplr(img2))  # 图像2左右翻转

    if circles1 is None or circles2 is None:
        return None

    pts1 = circles1[0][:, :2].astype(np.float32)
    pts2 = circles2[0][:, :2].astype(np.float32)

    tree = cKDTree(pts2)
    dists, indices = tree.query(pts1)

    dx_list = [pts2[j][0] - pt1[0] for pt1, j in zip(pts1, indices)]
    dy_list = [pts2[j][1] - pt1[1] for pt1, j in zip(pts1, indices)]

    if show_plot:
        h, w = img1.shape
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(norm1, cmap='gray')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.set_title("点匹配可视化")

        for pt1, j in zip(pts1, indices):
            pt2 = pts2[j]
            x0, y0 = pt1
            x1, y1 = pt2
            ax.plot(x0, y0, 'ro', markersize=5)
            ax.plot(x1, y1, 'bo', markersize=5)

        ax.legend(["图像1圆点", "图像2圆点（翻转）"], loc="upper right")
        plt.tight_layout()
        plt.show()

    return np.mean(dx_list), np.mean(dy_list)


def load_image_by_angle(angle, angle_list, file_list, proj_size):
    """
    加载最接近指定角度的投影图像
    """
    closest_idx = min(range(len(angle_list)), key=lambda i: abs(angle_list[i] - angle))
    return read_raw_image(file_list[closest_idx], *proj_size)


if __name__ == "__main__":
    # 配置参数
    data_dir = r"D:\\Data\\cbct\\CBCT0707"
    projection_size = [1420, 1420]
    spacing = 0.3  # mm
    angle_step = 30

    file_list, angle_list = read_projection_file(data_dir)
    angle_pairs = generate_symmetric_angle_pairs(angle_step)

    angle_centers = []
    u_offsets = []

    # 遍历角度对，估计偏移量
    for a1, a2 in angle_pairs:
        print(f"\n==== 处理角度对: {a1}° vs {a2}° ====")
        img1 = load_image_by_angle(a1, angle_list, file_list, projection_size)
        img2 = load_image_by_angle(a2, angle_list, file_list, projection_size)

        result = match_and_estimate_offset(img1, img2)
        if result is None:
            print("⚠️ 圆点未成功检测，跳过")
            continue

        dx, dy = result
        offset_mm = dx * spacing
        print(f"➡️ Δx = {dx:.2f} 像素 ({offset_mm:.2f} mm), Δy = {dy:.2f} 像素")

        angle_centers.append(a1)
        u_offsets.append(offset_mm)

    # 多项式拟合与可视化
    coeffs = np.polyfit(angle_centers, u_offsets, deg=2)
    fit_vals = np.polyval(coeffs, angle_centers)

    plt.figure(figsize=(10, 5))
    plt.plot(angle_centers, u_offsets, 'o-', label='测量值 (mm)')
    plt.plot(angle_centers, fit_vals, 'r--', label='拟合曲线')
    plt.xlabel("角度 (°)")
    plt.ylabel("Δx 偏移量 (mm)")
    plt.title("探测器偏移量 vs 角度")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 输出统计信息
    mean_offset = np.mean(u_offsets)
    std_offset = np.std(u_offsets)
    print(f"\n📊 平均横向偏移 Δx: {mean_offset:.3f} mm")
    print(f"📉 偏移标准差 Std: {std_offset:.3f} mm")
