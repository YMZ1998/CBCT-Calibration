import SimpleITK as sitk
import numpy as np


def invert_contrast_percentile(img, lower_pct=1, upper_pct=99):
    """
    使用 1%-99% 百分位范围进行强度反转，避免极端值干扰
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    p1 = np.percentile(arr, lower_pct)
    p99 = np.percentile(arr, upper_pct)

    # 裁剪 + 映射到 [0, 1]
    arr_clipped = np.clip(arr, p1, p99)
    arr_norm = (arr_clipped - p1) / (p99 - p1 + 1e-8)

    # 反转
    arr_inverted = 1.0 - arr_norm

    # 可映射回原范围（如果需要）
    # arr_inverted = arr_inverted * (p99 - p1) + p1

    out = sitk.GetImageFromArray(arr_inverted)
    out.CopyInformation(img)
    return out


if __name__ == '__main__':
    path = r"D:\debug\rigid_fixed.nii.gz"
    t2_img = sitk.ReadImage(path, sitk.sitkFloat32)

    t2_inverted = invert_contrast_percentile(t2_img)

    sitk.WriteImage(t2_inverted, path.replace(".nii.gz", "_inverted.nii.gz"))
