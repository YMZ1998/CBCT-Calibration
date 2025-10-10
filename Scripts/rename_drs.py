import os
import re


def rename_drs(src_path):
    for i in os.listdir(src_path):
        src = os.path.join(src_path, i)
        print(src)
        for p in os.listdir(src):
            file_path = os.path.join(src, p, 'ct.A.000.45.00.raw')
            file_path2 = os.path.join(src, p, 'A.raw')
            file_path3 = os.path.join(src, p, 'ct.B.000.45.00.raw')
            file_path4 = os.path.join(src, p, 'B.raw')

            if os.path.exists(file_path):
                print(file_path)

                os.rename(file_path, file_path2)
                os.rename(file_path3, file_path4)
                continue


def rename_files():
    target_angles = [a - b for a, b in zip([180, 0, 90], [45, 45, 45])]
    # target_angles = [a + b for a, b in zip([-90, 0, 90], [45, 45, 45])]
    print(target_angles)

    src_path = r'C:\Users\DATU\Documents\WeChat Files\wxid_hag56n8m9ejr22\FileStorage\File\2025-03\负负正B球馆'

    for folder_name in os.listdir(src_path):
        print(f"\n文件夹: {folder_name}")

        angles = []
        src_folder = os.path.join(src_path, folder_name)

        if not os.path.isdir(src_folder):
            continue

        for file_name in os.listdir(src_folder):
            match = re.search(r"([-+]?\d+\.\d+)\.raw", file_name)
            if match:
                angle = float(match.group(1))
                angles.append((angle, file_name))

        if not angles:
            print("未找到任何角度数据！")
            continue

        keep_files = set()
        for target in target_angles:
            closest_file = min(angles, key=lambda x: abs(x[0] - target))
            keep_files.add(closest_file[1])
            print(f"最接近 {target}° 的文件: {closest_file[1]} (角度: {closest_file[0]})")

        for angle, file_name in angles:
            if file_name not in keep_files:
                file_path = os.path.join(src_folder, file_name)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_name}")
                except Exception as e:
                    print(f"删除 {file_name} 失败: {e}")


if __name__ == "__main__":
    src_path = r'D:\Data\cbct\DR0707'

    rename_drs(src_path)
