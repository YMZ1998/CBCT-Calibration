import SimpleITK as sitk

# image = sitk.ReadImage(r"D:\Data\MIR\images\test\series_0.nii.gz")
image = sitk.ReadImage(r"C:\Users\DATU\Desktop\validation\fixed.nii.gz")
pixel_id = image.GetPixelIDTypeAsString()
print(f"SimpleITK 数据类型: {pixel_id}")
