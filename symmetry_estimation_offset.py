import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift

from utils import read_raw_image, read_projection_file, invert_image


def normalize_image(image):
    """标准化图像数据"""
    # return (image - np.mean(image)) / (np.std(image) + 1e-5)
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)


def compute_metrics(image1, image2, metric='ncc'):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    if metric == 'ncc':
        score = np.mean(image1 * image2)

    elif metric == 'mse':
        score = np.mean((image1 - image2) ** 2)

    elif metric == 'grad_ncc':
        grad1 = cv2.Sobel(image1, cv2.CV_32F, 1, 0, ksize=3)
        grad2 = cv2.Sobel(image2, cv2.CV_32F, 1, 0, ksize=3)
        mean1 = np.mean(grad1)
        mean2 = np.mean(grad2)
        numerator = np.sum((grad1 - mean1) * (grad2 - mean2))
        denominator = np.sqrt(np.sum((grad1 - mean1) ** 2) * np.sum((grad2 - mean2) ** 2))
        score = numerator / (denominator + 1e-8)

    elif metric == 'ssim':
        from skimage.metrics import structural_similarity as ssim
        # 将图像拉回 [0,1] 区间，因为 skimage 的 ssim 要求如此
        img1_norm = (image1 - image1.min()) / (image1.max() - image1.min() + 1e-8)
        img2_norm = (image2 - image2.min()) / (image2.max() - image2.min() + 1e-8)
        score = ssim(img1_norm, img2_norm, data_range=1.0)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return score


def estimate_u_offset(image_deg_0, image_deg_180, max_offset=50, metric='ncc', subpixel=True, subpixel_step=0.1,
                      show=False):
    """
    估计 image_deg_0 与 image_deg_180 左右翻转后的最佳横向 u 偏移（支持亚像素）
    支持 metric: 'ncc' 或 'mse'
    """
    image_deg_0 = normalize_image(image_deg_0.astype(np.float32))
    image_deg_180_flipped = normalize_image(np.fliplr(image_deg_180).astype(np.float32))

    h, w = image_deg_0.shape
    maximize = metric == 'ncc' or metric == 'grad_ncc' or metric == 'ssim'
    best_score = -np.inf if maximize else np.inf

    optimal_u_offset = 0
    matching_scores = []

    # === 粗搜索：整数偏移 ===
    print("正在执行整数优化...")
    for offset in range(-max_offset, max_offset + 1):
        if offset >= 0:
            region_0 = image_deg_0[:, offset:]
            region_180 = image_deg_180_flipped[:, :w - offset]
        else:
            region_0 = image_deg_0[:, :w + offset]
            region_180 = image_deg_180_flipped[:, -offset:]

        score = compute_metrics(region_0, region_180, metric=metric)
        if maximize:
            if score > best_score:
                best_score = score
                optimal_u_offset = offset
        elif metric == 'mse':
            if score < best_score:
                best_score = score
                optimal_u_offset = offset

        print(f"{metric} : best_score_fine: {best_score}, u 偏移: {offset:.2f} pixels")
        matching_scores.append(score)
    if show:
        visualize_matching_scores(matching_scores, optimal_u_offset, max_offset)

    sub_matching_scores = []
    # === 精搜索：亚像素插值匹配 ===
    if subpixel:
        print("正在执行亚像素优化...")
        sub_max_offset = 1.0
        sub_range = np.arange(optimal_u_offset - sub_max_offset, optimal_u_offset + sub_max_offset + 0.1, subpixel_step)
        best_score_fine = -np.inf if maximize else np.inf

        best_offset_fine = optimal_u_offset

        for offset_f in sub_range:
            # 使用 shift 进行亚像素精度平移（仅对180°翻转图做横向平移）
            shifted_180 = shift(image_deg_180_flipped, shift=(0, offset_f), order=1, mode='nearest')

            # 与原图裁剪到相同区域（避免边缘无效像素）
            crop = int(np.ceil(abs(offset_f)))
            region_0 = image_deg_0[:, crop:-crop] if crop > 0 else image_deg_0
            region_180 = shifted_180[:, crop:-crop] if crop > 0 else shifted_180

            if region_0.shape[1] < 10:
                continue  # 避免太小的对比区域

            score = compute_metrics(region_0, region_180, metric=metric)
            if maximize:
                if score > best_score_fine:
                    best_score_fine = score
                    best_offset_fine = offset_f
            elif metric == 'mse':
                if score < best_score_fine:
                    best_score_fine = score
                    best_offset_fine = offset_f

            print(f"{metric} : best_score_fine: {best_score_fine}, u 偏移: {offset_f:.2f} pixels")

            sub_matching_scores.append(score)
        if show:
            visualize_sub_matching_scores(sub_matching_scores, best_offset_fine, sub_range)

        optimal_u_offset = best_offset_fine
        print(f"✅ 亚像素优化后 u 偏移: {optimal_u_offset:.2f} pixels，匹配指标: {metric}")
    else:
        print(f"✅ 最佳整数 u 偏移: {optimal_u_offset} pixels，匹配指标: {metric}")
    return optimal_u_offset, matching_scores, sub_matching_scores


def visualize_projections(image_deg_0, image_deg_180):
    """显示 0°，180°，翻转后的 180° 图像"""
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.title("0°")
    plt.imshow(image_deg_0, cmap='gray')
    plt.subplot(1, 4, 2)
    plt.title("180°")
    plt.imshow(image_deg_180, cmap='gray')
    plt.subplot(1, 4, 3)
    plt.title("180° Flipped")
    plt.imshow(np.fliplr(image_deg_180), cmap='gray')
    plt.subplot(1, 4, 4)
    plt.imshow(np.fliplr(image_deg_180) - image_deg_0, cmap="jet")
    plt.colorbar()
    plt.title(f"Difference Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_matching_scores(matching_scores, optimal_offset, max_offset):
    """可视化匹配得分曲线"""
    offset_range = np.arange(-max_offset, max_offset + 1)
    plt.plot(offset_range, matching_scores)
    plt.axvline(optimal_offset, color='r', linestyle='--', label=f'Best Offset: {optimal_offset}')
    plt.title("Matching Score Curve")
    plt.xlabel("u Offset (pixels)")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_sub_matching_scores(matching_scores, optimal_offset, sub_range):
    """可视化匹配得分曲线"""
    plt.plot(sub_range, matching_scores)
    plt.axvline(optimal_offset, color='r', linestyle='--', label=f'Best Offset: {optimal_offset}')
    plt.title("Matching Score Curve")
    plt.xlabel("u Offset (pixels)")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_difference_before_after_alignment(image_deg_0, image_deg_180, optimal_offset, crop_margin=10):
    """
    显示对齐前后差值图（图像已标准化 + 翻转 + 支持亚像素平移）
    """
    # 预处理图像（标准化 + 翻转）
    image_deg_0 = normalize_image(image_deg_0.astype(np.float32))
    image_deg_180_flipped = normalize_image(np.fliplr(image_deg_180.astype(np.float32)))

    # 亚像素平移后的翻转图
    shifted_image_180 = shift(image_deg_180_flipped, shift=(0, optimal_offset), order=1, mode='nearest')

    # 计算裁剪范围（避免边界）
    h, w = image_deg_0.shape
    crop = max(abs(int(np.ceil(optimal_offset))), crop_margin)
    img0_crop = image_deg_0[:, crop:-crop]
    img180_crop_before = image_deg_180_flipped[:, crop:-crop]
    img180_crop_after = shifted_image_180[:, crop:-crop]

    # 差值计算
    diff_before = img0_crop - img180_crop_before
    diff_after = img0_crop - img180_crop_after

    # 显示两个差值图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(diff_before, cmap="jet")
    plt.colorbar()
    plt.title("Before Alignment")

    plt.subplot(1, 2, 2)
    plt.imshow(diff_after, cmap="jet")
    plt.colorbar()
    plt.title(f"After Alignment (u = {optimal_offset:.2f} px)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # === 参数设置 ===
    data_dir = r"D:\Data\cbct\CBCT0709"
    projection_size = [1420, 1420]
    spacing = 0.3
    target_angles = [0, 180]
    max_search_offset = 10
    metrics = ["ncc", "ssim", "mse", "grad_ncc"]
    match_metric = "grad_ncc"

    # === 读取投影文件名与角度 ===
    proj_file_list, angle_list = read_projection_file(data_dir)

    # === 寻找最接近目标角度的图像文件 ===
    selected_proj_files = []
    for target in target_angles:
        # target=target+2
        closest_idx = min(range(len(angle_list)), key=lambda i: abs(angle_list[i] - target))
        selected_proj_files.append(proj_file_list[closest_idx])
        print(f"最接近 {target}° 的投影: {proj_file_list[closest_idx]}（角度: {angle_list[closest_idx]}°）")

    # === 读取投影图像 ===
    image_deg_0 = read_raw_image(selected_proj_files[0], projection_size[0], projection_size[1])
    image_deg_180 = read_raw_image(selected_proj_files[1], projection_size[0], projection_size[1])

    image_deg_0 = invert_image(image_deg_0)
    image_deg_180 = invert_image(image_deg_180)
    # === 图像显示 ===
    # visualize_projections(image_deg_0, image_deg_180)

    # === 偏移估计 ===
    optimal_u_offset, matching_scores, sub_matching_scores = estimate_u_offset(
        image_deg_0, image_deg_180, max_offset=max_search_offset, metric=match_metric, show=True
    )

    # === 可视化差值图（用于对齐验证） ===
    visualize_difference_before_after_alignment(image_deg_0, image_deg_180, optimal_u_offset)

    print(f"u 偏移: {optimal_u_offset * spacing} mm")
    image_offset_mm = optimal_u_offset * spacing
    detector_shift_mm = image_offset_mm / 2
    print(f"实际探测器偏移 ≈ {detector_shift_mm:.3f} mm")
