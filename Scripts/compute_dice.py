import SimpleITK as sitk


def compute_dice(mask1_path, mask2_path):
    # 读取两个掩膜
    mask1 = sitk.ReadImage(mask1_path, sitk.sitkUInt8)
    mask2 = sitk.ReadImage(mask2_path, sitk.sitkUInt8)

    # 确保掩膜维度和空间信息一致
    if mask1.GetSize() != mask2.GetSize():
        raise ValueError("Mask sizes do not match.")

    # 将图像转为 numpy 数组
    mask1_array = sitk.GetArrayFromImage(mask1)
    mask2_array = sitk.GetArrayFromImage(mask2)

    # 计算 Dice 系数
    intersection = (mask1_array & mask2_array).sum()
    volume_sum = mask1_array.sum() + mask2_array.sum()

    if volume_sum == 0:
        return 1.0  # 两个都是空的，定义 Dice = 1
    dice = 2.0 * intersection / volume_sum
    return dice


def compute_dice_sitk(mask1_path, mask2_path):
    mask1 = sitk.ReadImage(mask1_path, sitk.sitkUInt8)
    mask2 = sitk.ReadImage(mask2_path, sitk.sitkUInt8)

    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(mask1, mask2)
    return overlap_filter.GetDiceCoefficient()


output_path1 = r'D:\Data\MIR\images\mask_cube.nii.gz'
output_path2 = r"C:\Users\DATU\Desktop\validation\warped_mask.nii.gz"
output_path3 = r'D:\Data\MIR\images\mask_cylinder.nii.gz'
dice_score = compute_dice(output_path1, output_path2)
dice_score2 = compute_dice_sitk(output_path1, output_path3)
print("Dice coefficient:", dice_score)
print("Dice coefficient:", dice_score2)
