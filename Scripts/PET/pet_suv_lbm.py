import os
import pathlib as plb
import shutil
import tempfile

import dicom2nifti
import pydicom

from Scripts.PET.pet_suv2 import convert_pet_to_suv, conv_time


def calculate_lbm(weight, height, sex='M', method='janma'):
    """
    根据体重(kg)、身高(cm)、性别计算 Lean Body Mass (kg)
    method: 'james' 或 'janma'
    """
    if method.lower() == 'james':
        if sex.upper() == 'M':
            lbm = 1.1 * weight - 128 * (weight / height) ** 2
        else:
            lbm = 1.07 * weight - 148 * (weight / height) ** 2
    else:  # janmahasatian
        height_m = height / 100
        BMI = weight / (height_m ** 2)
        print(f"BMI: {BMI:.2f}")
        if sex.upper() == 'M':
            lbm = 9270 * weight / (6680 + 216 * BMI)
        else:
            lbm = 9270 * weight / (8780 + 244 * BMI)
    return lbm


def calculate_suv_factor(dcm_path, use_lbm=False, height=None, sex='M'):
    """根据 PET DICOM 文件计算 SUV 转换系数，可选使用 LBM"""
    ds = pydicom.dcmread(str(dcm_path))

    total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    acq_time = ds.AcquisitionTime
    weight = ds.PatientWeight

    # 如果使用 LBM，必须提供身高
    if use_lbm:
        if height is None:
            raise ValueError("使用 LBM 时必须提供患者身高 (cm)")
        weight_for_suv = calculate_lbm(weight, height, sex)
    else:
        weight_for_suv = weight

    # 衰变校正
    time_diff = conv_time(acq_time) - conv_time(start_time)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)
    suv_factor = 1000 * weight_for_suv / act_dose

    print(f"✅ Time difference: {time_diff:.2f} s")
    print(f"✅ RadionuclideHalfLife: {half_life:.2f} s")
    print(f"✅ weight used: {weight:.2f} kg")
    print(f"✅ height used: {height:.2f} cm")
    print(f"✅ Decay-corrected dose: {act_dose:.2e} Bq")
    print(f"✅ SUV factor: {suv_factor:.6f}")
    return suv_factor


def convert_pet_dicom_to_nifti(dicom_folder, output_folder, use_lbm=False, height=None, sex='M'):
    """将 PET DICOM 转换为 NIfTI 并计算 SUV，可选使用 LBM"""
    dicom_folder = plb.Path(dicom_folder)
    output_folder = plb.Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    first_pet = next(dicom_folder.glob('*.dcm'))
    ds = pydicom.dcmread(str(first_pet))
    if ds.Modality not in ['PT', 'PET']:
        raise ValueError("❌ 输入目录不是 PET 模态")

    # 计算 SUV 因子
    suv_factor = calculate_suv_factor(first_pet, use_lbm=use_lbm, height=height, sex=sex)

    # 转换为 NIfTI
    print("\n🔹 Converting PET DICOM to NIfTI...")
    with tempfile.TemporaryDirectory() as tmp:
        dicom2nifti.convert_directory(dicom_folder, tmp, compression=True, reorient=True)
        nii_file = next(plb.Path(tmp).glob('*.nii.gz'))
        shutil.copy(nii_file, output_folder / 'PET.nii.gz')

    print(f"✅ Saved PET image: {output_folder / 'PET.nii.gz'}")

    # 计算 SUV
    convert_pet_to_suv(output_folder / 'PET.nii.gz', suv_factor)

    print("\n🎯 PET SUV conversion complete!")
    print("Output folder:", output_folder)


if __name__ == "__main__":
    dicom_folder = r"D:\Data\pet\Sample Data\PET"
    save_folder = r"D:\Data\pet"

    # 如果想用 LBM 计算，需要提供身高和性别
    convert_pet_dicom_to_nifti(dicom_folder, save_folder, use_lbm=True, height=175, sex='M')
