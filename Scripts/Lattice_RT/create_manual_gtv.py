import os
import numpy as np
import SimpleITK as sitk


def array_to_sitk(arr, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def create_manual_gtv(shape=(64, 64, 64), spacing=(2.0, 2.0, 2.0), type='sphere'):
    sx, sy, sz = shape
    arr = np.zeros((sz, sy, sx), dtype=np.uint8)

    if type == 'sphere':
        center = np.array([sz // 2, sy // 2, sx // 2])

        # 物理半径 (mm)
        radius_mm = min(sz * spacing[2], sy * spacing[1], sx * spacing[0]) / 3.0

        # voxel 坐标转换为物理坐标 (mm)
        zz, yy, xx = np.meshgrid(np.arange(sz), np.arange(sy), np.arange(sx), indexing='ij')
        px = (xx - center[2]) * spacing[0]
        py = (yy - center[1]) * spacing[1]
        pz = (zz - center[0]) * spacing[2]

        dist2 = px ** 2 + py ** 2 + pz ** 2
        arr[dist2 <= radius_mm ** 2] = 1

    elif type == 'cube':
        arr[sz // 4:3 * sz // 4, sy // 4:3 * sy // 4, sx // 4:3 * sx // 4] = 1

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


if __name__ == "__main__":
    out_path = r"D:\debug\gtv_manual.nii.gz"

    gtv_img = create_manual_gtv(shape=(256, 256, 128), spacing=(1.0, 1.0, 1.0))
    sitk.WriteImage(gtv_img, out_path)
    print("手动 GTV mask 已保存:", out_path)
