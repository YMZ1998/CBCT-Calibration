import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


def plot_suv_histogram(pet_file, roi_file=None, voxel_size_mm=2.0, peak_size_cm3=1.0, bins=100, suv_threshold=None):
    """
    绘制 PET SUV 直方图 + CDF，并标注 SUVmax、SUVmean、SUVpeak，可选显示阈值高亮。

    参数:
        pet_file (str): PET NIfTI 文件路径
        roi_file (str, optional): ROI mask 文件路径，只分析 ROI 区域
        voxel_size_mm (float): PET voxel 尺寸（假设各向同性）
        peak_size_cm3 (float): 计算 SUVpeak 区域体积（cm³）
        bins (int): 直方图 bins 数量
        suv_threshold (float, optional): 阈值 SUV，高于此值的 voxel 可高亮并计算占比
    """
    # =========================
    # 1. 读取 PET 图像
    # =========================
    pet_img = nib.load(pet_file)
    pet_data = pet_img.get_fdata()

    # =========================
    # 2. 可选：提取 ROI
    # =========================
    if roi_file is not None:
        roi_mask = nib.load(roi_file).get_fdata()
        pet_data = pet_data[roi_mask > 0]

    # =========================
    # 3. 去掉背景或零值
    # =========================
    pet_data_flat = pet_data.flatten()
    pet_data_flat = pet_data_flat[pet_data_flat > 0]

    # =========================
    # 4. 计算指标
    # =========================
    SUVmax = pet_data_flat.max()
    SUVmean = pet_data_flat.mean()

    # 计算 SUVpeak（1cm³ 区域平均最大）
    peak_size_mm3 = peak_size_cm3 * 1000.0  # cm³ -> mm³
    kernel_size = int(round((peak_size_mm3 / voxel_size_mm ** 3) ** (1 / 3)))
    smoothed = uniform_filter(pet_data, size=kernel_size)
    SUVpeak = smoothed.max()

    # =========================
    # 5. 绘制直方图 + CDF + 标注
    # =========================
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # 直方图
    counts, bins_edges, patches = ax1.hist(pet_data_flat, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('SUV')
    ax1.set_ylabel('Voxel Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 阈值高亮
    if suv_threshold is not None:
        for patch, left_edge in zip(patches, bins_edges[:-1]):
            if left_edge >= suv_threshold:
                patch.set_facecolor('salmon')
        # 计算占比
        fraction_above = np.sum(pet_data_flat >= suv_threshold) / len(pet_data_flat)
        print(f"SUV ≥ {suv_threshold} 占比: {fraction_above * 100:.2f}%")

    # 累积分布曲线
    ax2 = ax1.twinx()
    cdf = np.cumsum(counts) / np.sum(counts)
    ax2.plot(bins_edges[:-1], cdf, color='red', label='CDF')
    ax2.set_ylabel('Cumulative Fraction', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # 标注 SUVmax / mean / peak
    for suv, label in zip([SUVmax, SUVmean, SUVpeak], ['SUVmax', 'SUVmean', 'SUVpeak']):
        ax1.axvline(suv, color='green', linestyle='--')
        ax1.text(suv, max(counts) * 0.9, f'{label}\n{suv:.2f}', rotation=90, color='green', va='top', ha='right')

    plt.title('SUV Histogram with CDF and Key Metrics')
    plt.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    pet_file = r"D:\Data\pet\SUV2.nii.gz"
    roi_file = None  # 如果有 ROI mask 可设置路径
    suv_threshold = 2.5

    plot_suv_histogram(pet_file, roi_file=roi_file, suv_threshold=suv_threshold)
