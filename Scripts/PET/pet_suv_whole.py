import os
import pathlib as plb
import tempfile
import shutil
import numpy as np
import nibabel as nib
import dicom2nifti
import pydicom


def conv_time(time_str):
    """将 DICOM 时间字符串 (HHMMSS.FFFFFF) 转换为秒"""
    return (float(time_str[:2]) * 3600 +
            float(time_str[2:4]) * 60 +
            float(time_str[4:]))


def calculate_lbm(weight, height, sex='M', method='janma'):
    """根据体重(kg)、身高(cm)、性别计算 Lean Body Mass (kg)"""
    if method.lower() == 'james':
        if sex.upper() == 'M':
            lbm = 1.1 * weight - 128 * (weight / height) ** 2
        else:
            lbm = 1.07 * weight - 148 * (weight / height) ** 2
    else:  # janmahasatian
        height_m = height / 100
        BMI = weight / (height_m ** 2)
        if sex.upper() == 'M':
            lbm = 9270 * weight / (6680 + 216 * BMI)
        else:
            lbm = 9270 * weight / (8780 + 244 * BMI)
    return lbm


def calculate_bsa(weight, height, method='duBois'):
    """根据体重(kg)、身高(cm)计算体表面积 BSA (m²)"""
    if method.lower() == 'dubois':
        bsa = 0.007184 * (weight ** 0.425) * (height ** 0.725)
    else:
        # 其他公式可扩展
        bsa = 0.20247 * (height ** 0.725) * (weight ** 0.425)
    return bsa


def calculate_suv_factors(dcm_path, height=None, sex='M'):
    """计算三种 SUV 因子：BW、LBM、BSA"""
    ds = pydicom.dcmread(str(dcm_path))
    total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    acq_time = ds.AcquisitionTime
    weight = ds.PatientWeight

    # 衰变校正
    time_diff = conv_time(acq_time) - conv_time(start_time)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)

    # SUVbw
    suv_bw = 1000 * weight / act_dose
    print(f"Weight: {weight:.2f} kg")

    # SUVlbm
    lbm = calculate_lbm(weight, height, sex)
    print(f"LBM: {lbm:.2f} kg")
    suv_lbm = 1000 * lbm / act_dose

    # SUVbsa
    bsa = calculate_bsa(weight, height)
    print(f"BSA: {bsa:.2f} m²")
    suv_bsa = 1000 * bsa / act_dose

    print(f"✅ Decay-corrected dose: {act_dose:.2e} Bq")
    print(f"✅ SUV_BW factor: {suv_bw:.6f}")
    print(f"✅ SUV_LBM factor: {suv_lbm:.6f}")
    print(f"✅ SUV_BSA factor: {suv_bsa:.6f}")

    return suv_bw, suv_lbm, suv_bsa


def convert_pet_to_suv(pet_nii_path, suv_factors):
    """生成三种 SUV 图像"""
    pet = nib.load(str(pet_nii_path))
    affine = pet.affine
    pet_data = pet.get_fdata()

    print(np.min(pet_data), np.max(pet_data))

    names = ['BW', 'LBM', 'BSA']
    out_paths = []
    for factor, name in zip(suv_factors, names):
        suv_data = (pet_data * factor).astype(np.float32)
        out_file = pet_nii_path.parent / f"SUV_{name}.nii.gz"
        suv_img = nib.Nifti1Image(suv_data, affine)
        print(f"SUV_{name} factor: {factor:.6f}")
        print("SUV 图像 shape:", suv_img.shape)
        print("SUV 最大值:", np.max(suv_img.get_fdata()))
        print("SUV 平均值:", np.mean(suv_img.get_fdata()))
        nib.save(suv_img, out_file)
        print(f"✅ Saved SUV_{name}: {out_file}")
        out_paths.append(out_file)
    return out_paths


def convert_pet_dicom_to_nifti(dicom_folder, output_folder, height, sex='M'):
    """将 PET DICOM 转换为 NIfTI 并生成三种 SUV 图像"""
    dicom_folder = plb.Path(dicom_folder)
    output_folder = plb.Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    first_pet = next(dicom_folder.glob('*.dcm'))
    ds = pydicom.dcmread(str(first_pet))
    if ds.Modality not in ['PT', 'PET']:
        raise ValueError("❌ 输入目录不是 PET 模态")

    # 计算 SUV 因子
    suv_factors = calculate_suv_factors(first_pet, height=height, sex=sex)

    # 转换为 NIfTI
    print("\n🔹 Converting PET DICOM to NIfTI...")
    with tempfile.TemporaryDirectory() as tmp:
        dicom2nifti.convert_directory(dicom_folder, tmp, compression=True, reorient=True)
        nii_file = next(plb.Path(tmp).glob('*.nii.gz'))
        shutil.copy(nii_file, output_folder / 'PET.nii.gz')
    print(f"✅ Saved PET image: {output_folder / 'PET.nii.gz'}")

    # 生成 SUV 图像
    convert_pet_to_suv(output_folder / 'PET.nii.gz', suv_factors)

    print("\n🎯 PET SUV conversion complete!")
    print("Output folder:", output_folder)


if __name__ == "__main__":
    dicom_folder = r"D:\Data\pet\Sample Data\PET"
    save_folder = r"D:\Data\pet"
    patient_height = 175  # cm
    patient_sex = 'F'

    convert_pet_dicom_to_nifti(dicom_folder, save_folder, height=patient_height, sex=patient_sex)
