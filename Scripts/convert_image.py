import SimpleITK as sitk

case_path1=r'D:\debug\moving_vol_roi.nii.gz'
case_path2=r'D:\debug\fixed_vol_roi.nii.gz'
def prcocess(case_path):
    image = sitk.ReadImage(case_path)
    image_uint8 = sitk.Cast(image, sitk.sitkUInt8)

    sitk.WriteImage(image_uint8, case_path)

prcocess(case_path1)
prcocess(case_path2)
