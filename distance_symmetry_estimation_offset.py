from process_raw import detect_circles
from symmetry_estimation_offset import visualize_projections
from utils import read_projection_file, read_raw_image, invert_image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # === 参数设置 ===
    data_dir = r"D:\Data\cbct\CBCT0707"
    projection_size = [1420, 1420]
    target_angles = [0, 180]

    # === 读取投影文件名与角度 ===
    proj_file_list, angle_list = read_projection_file(data_dir)

    # === 寻找最接近目标角度的图像文件 ===
    selected_proj_files = []
    for target in target_angles:
        closest_idx = min(range(len(angle_list)), key=lambda i: abs(angle_list[i] - target))
        selected_proj_files.append(proj_file_list[closest_idx])
        print(f"最接近 {target}° 的投影: {proj_file_list[closest_idx]}（角度: {angle_list[closest_idx]}°）")

    # === 读取并预处理图像 ===
    image_0 = read_raw_image(selected_proj_files[0], *projection_size)
    image_180 = read_raw_image(selected_proj_files[1], *projection_size)
    image_180_flip = np.fliplr(image_180)

    # visualize_projections(image_0, image_180_flip)  # 可选：显示原始图

    # === 执行圆检测 ===
    circles_0, output_0, norm_0 = detect_circles(image_0)
    circles_180, output_180, norm_180 = detect_circles(image_180_flip)

    # === 可视化三张图：两个原图 + 圆心对比图（无底图）===
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 显示0°图像带圆
    axs[0].imshow(output_0[..., ::-1])
    axs[0].set_title("Detected Circles - 0°")
    axs[0].axis("off")

    # 显示180°翻转图像带圆
    axs[1].imshow(output_180[..., ::-1])
    axs[1].set_title("Detected Circles - 180° (Flipped)")
    axs[1].axis("off")

    from scipy.spatial import cKDTree  # 用于快速最近邻搜索

    # === 空白图显示匹配结果（最近邻 + 距离） ===
    axs[2].set_title("Nearest Point Matching with Distances")
    axs[2].set_xlim([0, projection_size[0]])
    axs[2].set_ylim([projection_size[1], 0])
    axs[2].axis("off")
    axs[2].imshow(norm_0)
    if circles_0 is not None and circles_180 is not None:
        pts_0 = circles_0[0][:, :2]  # (N, 2)
        pts_180 = circles_180[0][:, :2]  # (M, 2)

        tree_180 = cKDTree(pts_180)
        dists, indices = tree_180.query(pts_0, k=1)  # 找到每个 0° 点在 180° 中的最近邻

        for i, (pt0, idx_180) in enumerate(zip(pts_0, indices)):
            pt180 = pts_180[idx_180]
            pt0 = pt0.astype(np.float32)
            pt180 = pt180.astype(np.float32)
            dist = np.linalg.norm(pt0 - pt180)
            print(f"{i}: {dist:.1f}")

            x0, y0 = pt0
            x1, y1 = pt180
            print(pt0, pt180)

            diff_x = x1 - x0
            diff_y = y1 - y0
            # 绘制点
            axs[2].plot(x0, y0, 'ro', markersize=4)
            axs[2].plot(x1, y1, 'bo', markersize=4)

            # 连线
            axs[2].plot([x0, x1], [y0, y1], 'k--', linewidth=1)

            # 标注距离
            mid_x, mid_y = (x0 + x1) / 2 + 120, (y0 + y1) / 2
            # axs[2].text(mid_x, mid_y, f"{dist:.1f},({diff_x},{diff_y})", color='black', fontsize=8)
            axs[2].text(mid_x, mid_y, f"{dist:.1f}\n({diff_x:.1f}, {diff_y:.1f})",
                        color='black', fontsize=8, ha='center', va='center')

        axs[2].legend(['0° Circles', '180° Flipped Circles', 'Distance'])

    else:
        axs[2].text(0.5, 0.5, "圆点未成功检测", ha='center', va='center', transform=axs[2].transAxes)

    plt.tight_layout()
    plt.show()

