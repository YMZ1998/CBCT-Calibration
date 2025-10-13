import os
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ---------------- 默认参考值 ----------------
DEFAULT_DOSE = 1.83e8  # 370 MBq
DEFAULT_HALF_LIFE = 6586  # 18F 半衰期 (s)
DEFAULT_INJECT_TIME = "154009"  # 注射时间 HHMMSS
DEFAULT_PATIENT_WEIGHT = 60.0  # kg


# ---------------- 工具函数 ----------------
def time_to_seconds(timestr):
    h = int(timestr[0:2])
    m = int(timestr[2:4])
    s = float(timestr[4:])
    return h * 3600 + m * 60 + s


# ---------------- 扫描 PET 序列 ----------------
def find_pet_series(folder):
    series_dict = {}
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                modality = getattr(ds, 'Modality', '')
                series_desc = getattr(ds, 'SeriesDescription', '')
                series_id = getattr(ds, 'SeriesInstanceUID', '')
                if modality == 'PT':
                    series_dict[series_id] = series_desc
    return series_dict


# ---------------- 读取 DICOM ----------------
def load_pet_dicom_series(folder_path, series_id):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path, series_id)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, r"D:\Data\pet\\pet.nii.gz")
    image_np = sitk.GetArrayFromImage(image)

    ds = pydicom.dcmread(dicom_names[0], stop_before_pixels=True)

    # 重缩放
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))

    # 患者体重
    patient_weight = getattr(ds, 'PatientWeight', DEFAULT_PATIENT_WEIGHT)

    # 放射性药物信息
    rads_seq = getattr(ds, 'RadiopharmaceuticalInformationSequence', None)
    if rads_seq is not None and len(rads_seq) > 0:
        rads = rads_seq[0]
        dose = getattr(rads, 'RadionuclideTotalDose', DEFAULT_DOSE)
        half_life = getattr(rads, 'RadionuclideHalfLife', DEFAULT_HALF_LIFE)
        inject_time = getattr(rads, 'RadiopharmaceuticalStartTime', DEFAULT_INJECT_TIME)
    else:
        dose = DEFAULT_DOSE
        half_life = DEFAULT_HALF_LIFE
        inject_time = DEFAULT_INJECT_TIME

    # 扫描时间
    scan_time = getattr(ds, 'SeriesTime', None)
    # print("AcquisitionTime time:", ds.AcquisitionTime)
    # print("SeriesTime time:", ds.SeriesTime)
    if scan_time is None:
        scan_time = DEFAULT_INJECT_TIME  # 默认同注射时间
    print("Scan time:", scan_time)

    info = {
        'RescaleSlope': slope,
        'RescaleIntercept': intercept,
        'PatientWeight': patient_weight,
        'RadionuclideTotalDose': dose,
        'RadionuclideHalfLife': half_life,
        'RadiopharmaceuticalStartTime': inject_time,
        'AcquisitionTime': scan_time
    }
    print("info: ", info)
    return image_np, info

def conv_time(time_str):
    # function for time conversion in DICOM tag
    return (float(time_str[:2]) * 3600 + float(time_str[2:4]) * 60 + float(time_str[4:13]))

def calculate_suv2(image_np, info):
    total_dose = info['RadionuclideTotalDose']
    start_time = info['RadiopharmaceuticalStartTime']
    half_life = info['RadionuclideHalfLife']
    acq_time = info['AcquisitionTime']
    weight = info['PatientWeight']
    time_diff = conv_time(acq_time) - conv_time(start_time)
    print("Time difference:", time_diff)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)
    suv_factor = 1000 * weight / act_dose
    return suv_factor * image_np


# ---------------- SUV 计算 ----------------
def calculate_suv(image_np, info):
    C_voxel = image_np * info['RescaleSlope'] #+ info['RescaleIntercept']
    delta_t = time_to_seconds(info['AcquisitionTime']) - time_to_seconds(info['RadiopharmaceuticalStartTime'])
    print("Delta t:", delta_t)
    lambda_decay = np.log(2) / info['RadionuclideHalfLife']
    A_inj_corr = info['RadionuclideTotalDose'] * np.exp(-lambda_decay * delta_t)
    suv = C_voxel * info['PatientWeight'] / A_inj_corr*1000
    return suv


# ---------------- 可视化 ----------------
def visualize_suv(suv_image):
    mid_z = suv_image.shape[0] // 2
    max_proj = np.max(suv_image, axis=0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(suv_image[mid_z], cmap='hot')
    plt.colorbar()
    plt.title("Mid-axis SUV")
    plt.subplot(1, 2, 2)
    plt.imshow(max_proj, cmap='hot')
    plt.colorbar()
    plt.title("Maximum projection SUV")
    plt.show()


# ---------------- 保存 ----------------
def save_suv(suv_image, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # NIfTI
    suv_sitk = sitk.GetImageFromArray(suv_image)
    nifti_path = os.path.join(save_dir, "SUV.nii.gz")
    sitk.WriteImage(suv_sitk, nifti_path)


# ---------------- 主流程 ----------------
if __name__ == "__main__":
    dicom_folder = r"D:\Data\pet\33298_HELIQIANG"
    # dicom_folder = r"D:\Data\pet\Sample Data\PET"
    save_folder = r"D:\Data\pet"

    # 1. 查找 PET 序列
    pet_series = find_pet_series(dicom_folder)
    if not pet_series:
        print("未找到 PET 序列")
        exit()

    print("找到的 PET 序列：")
    for i, (sid, desc) in enumerate(pet_series.items()):
        print(f"{i}: {desc} (SeriesID: {sid})")

    # 2. 用户选择序列
    # choice = int(input("请选择序列编号: "))
    series_id = list(pet_series.keys())[0]

    # 3. 读取 DICOM
    pet_image, info = load_pet_dicom_series(dicom_folder, series_id)

    # 4. 计算 SUV
    suv_image = calculate_suv2(pet_image, info)

    # 5. 输出统计信息
    print("SUV 图像 shape:", suv_image.shape)
    print("SUV 最大值:", np.max(suv_image))
    print("SUV 平均值:", np.mean(suv_image))

    # 6. 保存结果
    save_suv(suv_image, save_folder)

    # 7. 可视化
    visualize_suv(suv_image)
