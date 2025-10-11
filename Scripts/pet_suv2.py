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


def calculate_suv_factor(dcm_path):
    """根据 PET DICOM 文件计算 SUV 转换系数"""
    ds = pydicom.dcmread(str(dcm_path))

    total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    acq_time = ds.AcquisitionTime
    weight = ds.PatientWeight

    # 衰变校正
    time_diff = conv_time(acq_time) - conv_time(start_time)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)
    suv_factor = 1000 * weight / act_dose

    print(f"✅ Time difference: {time_diff:.2f} s")
    print(f"✅ RadionuclideHalfLife: {half_life:.2f} s")
    print(f"✅ weight: {weight} kg")
    print(f"✅ Decay-corrected dose: {act_dose:.2e} Bq")
    print(f"✅ SUV factor: {suv_factor:.6f}")
    return suv_factor


def convert_pet_to_suv(pet_nii_path, suv_factor):
    """将 PET NIfTI 图像转换为 SUV"""
    pet = nib.load(str(pet_nii_path))
    affine = pet.affine
    pet_data = pet.get_fdata()
    pet_suv_data = (pet_data * suv_factor).astype(np.float32)
    suv_img = nib.Nifti1Image(pet_suv_data, affine)
    out_path = pet_nii_path.parent / "SUV.nii.gz"
    nib.save(suv_img, out_path)
    print(f"✅ Saved SUV image: {out_path}")
    return out_path


def convert_pet_dicom_to_nifti(dicom_folder, output_folder):
    """将 PET DICOM 转换为 NIfTI 并计算 SUV"""
    dicom_folder = plb.Path(dicom_folder)
    output_folder = plb.Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    first_pet = next(dicom_folder.glob('*.dcm'))
    ds = pydicom.dcmread(str(first_pet))
    if ds.Modality not in ['PT', 'PET']:
        raise ValueError("❌ 输入目录不是 PET 模态")

    # 计算 SUV 因子
    suv_factor = calculate_suv_factor(first_pet)

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

    convert_pet_dicom_to_nifti(dicom_folder, save_folder)
