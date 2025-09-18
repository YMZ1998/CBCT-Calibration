import numpy as np
import matplotlib.pyplot as plt

from symmetry_estimation_offset import estimate_u_offset, clip_image, visualize_difference_before_after_alignment

scale = 900.07 / 1451.42


def generate_symmetric_angle_pairs(angle_step=30):
    """
    生成 [-180, 180) 范围内的对称角度对，例如：
    (-150, 30), (-120, 60), ..., (150, -30)
    """
    angle_pairs = []
    for a in range(0, 360, angle_step):
        a1 = a
        a2 = a + 180
        if a2 > 360:
            continue
        angle_pairs.append((a1, a2))
    return angle_pairs


def estimate_u_offsets_for_symmetric_angles(
    data_dir,
    projection_size,
    spacing=0.3,
    angle_step=30,
    max_offset=10,
    metric='ssim'
):
    from utils import read_projection_file, read_raw_image, invert_image

    proj_file_list, angle_list = read_projection_file(data_dir)
    angle_pairs = generate_symmetric_angle_pairs(angle_step)
    print("📐 对称角度对:", angle_pairs)

    offset_data = []

    for angle_src, angle_mirror in angle_pairs:
        print(f"🔍 正在处理 {angle_src:.1f}° vs {angle_mirror:.1f}°")

        idx_src = min(range(len(angle_list)), key=lambda i: abs(angle_list[i] - angle_src))
        idx_mirror = min(range(len(angle_list)), key=lambda i: abs(angle_list[i] - angle_mirror))

        img_src = read_raw_image(proj_file_list[idx_src], *projection_size)
        img_mirror = read_raw_image(proj_file_list[idx_mirror], *projection_size)

        img_src, img_mirror = clip_image(img_src, img_mirror)

        img_src = invert_image(img_src)
        img_mirror = invert_image(img_mirror)

        offset, *_ = estimate_u_offset(
            img_src, img_mirror, max_offset=max_offset, metric=metric, show=False)

        # === 可视化差值图（用于对齐验证） ===
        # visualize_difference_before_after_alignment(img_src, img_mirror, offset)

        offset_mm = offset * spacing / 2 * scale

        offset_data.append({
            "angle_src": angle_list[idx_src],
            "angle_mirror": angle_list[idx_mirror],
            "offset_px": offset,
            "offset_mm": offset_mm
        })

        print(f"✅ {angle_list[idx_src]:.1f}° vs {angle_list[idx_mirror]:.1f}° → u = {offset:.2f}px → {offset_mm:.2f}mm")

    return offset_data


def plot_u_offsets_vs_angle(offset_data):
    angles = [r['angle_src'] for r in offset_data]
    offsets_mm = [r['offset_mm'] for r in offset_data]

    # --- 去除最高和最低值 ---
    offsets_arr = np.array(offsets_mm)
    angles_arr = np.array(angles)

    # 找到最大和最小的索引
    max_idx = np.argmax(offsets_arr)
    min_idx = np.argmin(offsets_arr)

    # 保留其余数据
    mask = np.ones_like(offsets_arr, dtype=bool)
    mask[[max_idx, min_idx]] = False

    filtered_offsets = offsets_arr[mask]
    filtered_angles = angles_arr[mask]

    # 重新计算均值和标准差
    mean_offset = np.mean(filtered_offsets)
    std_offset = np.std(filtered_offsets)

    print(f"\n📊 平均 u 偏移(去极值): {mean_offset:.3f} mm，标准差: {std_offset:.3f} mm")
    detector_shift_mm = mean_offset
    print(f"实际探测器偏移(去极值) ≈ {detector_shift_mm:.3f} mm")

    # 多项式拟合也用去除后的数据
    coeffs = np.polyfit(filtered_angles, filtered_offsets, deg=2)
    fit_vals = np.polyval(coeffs, filtered_angles)

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(angles, offsets_mm, 'o-', label='Measured (all)')
    plt.plot(filtered_angles, filtered_offsets, 'go', label='Filtered')  # 剩余数据
    plt.plot(filtered_angles, fit_vals, 'r--', label='Poly Fit (filtered)')
    plt.xlabel("Angle (deg)")
    plt.ylabel("u Offset (mm)")
    plt.grid(True)
    plt.legend()
    plt.title("u Offset vs Projection Angle")
    plt.tight_layout()
    plt.savefig("../result/u_offset_vs_angle.png")
    plt.show()


if __name__ == "__main__":
    # data_dir = r"D:\Data\cbct\CBCT0703"
    data_dir = r"D:\Data\cbct\CBCT0331\A"
    projection_size = [1420, 1420]
    spacing = 0.3
    angle_step = 30
    metric = 'grad_ncc'

    offset_data = estimate_u_offsets_for_symmetric_angles(
        data_dir, projection_size, spacing=spacing,
        angle_step=angle_step, max_offset=10, metric=metric
    )

    plot_u_offsets_vs_angle(offset_data)
