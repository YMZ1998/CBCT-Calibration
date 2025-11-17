import os
import numpy as np
import SimpleITK as sitk


def array_to_sitk(arr, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def create_manual_gtv(shape=(64, 64, 64), spacing=(1.0, 1.0, 1.0)):
    """
    shape: (z,y,x)
    spacing: voxel spacing (sx, sy, sz)
    返回: SimpleITK 图像，二值 mask
    """
    sz, sy, sx = shape
    gtv_arr = np.zeros(shape, dtype=np.uint8)

    # 示例 1：在中心创建一个球形 GTV
    center = np.array([sz // 2, sy // 2, sx // 2])
    radius_vox = min(sz, sy, sx) // 4  # 半径 = 体积最小边/4

    zz, yy, xx = np.meshgrid(np.arange(sz), np.arange(sy), np.arange(sx), indexing="ij")
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    gtv_arr[dist2 <= radius_vox ** 2] = 1

    gtv_img = array_to_sitk(gtv_arr, spacing=spacing)
    return gtv_img


if __name__ == "__main__":
    out_path = r"D:\debug\gtv_manual.nii.gz"

    gtv_img = create_manual_gtv(shape=(256, 256, 256), spacing=(1.0, 1.0, 1.0))
    sitk.WriteImage(gtv_img, out_path)
    print("手动 GTV mask 已保存:", out_path)
