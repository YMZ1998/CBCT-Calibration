import numpy as np
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk


def shrink_mask(mask_arr, spacing, shrink_mm):
    """
    mask_arr : 3D numpy array (0/1)
    spacing  : (sz, sy, sx)   # 来自 SimpleITK：img.GetSpacing()
    shrink_mm: 要收缩的物理距离（mm）
    """

    # 距离变换，mask 内部的正距离
    dt = distance_transform_edt(mask_arr > 0, sampling=spacing)
    print(dt.min(), dt.max())

    # 内缩 = 保留距离 >= shrink_mm 的体素
    shrunk = (dt >= shrink_mm).astype(np.uint8)

    return shrunk


if __name__ == "__main__":
    mask_img = sitk.ReadImage(r"D:\\Data\\cbct\\CT2\\skull_mask.mhd")
    mask_arr = sitk.GetArrayFromImage(mask_img)
    spacing = mask_img.GetSpacing()  # (sx, sy, sz)

    shrink_mm = 5.0
    shrunk_arr = shrink_mask(mask_arr, spacing[::-1], shrink_mm)

    new_img = sitk.GetImageFromArray(shrunk_arr)
    new_img.CopyInformation(mask_img)
    sitk.WriteImage(new_img, r"D:\debug\mask_shrink.nii.gz")
