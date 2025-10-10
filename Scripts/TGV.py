import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

def tgv_denoise_3d_torch(volume, lambda_tgv=0.1, alpha0=1.0, alpha1=1.2, max_iter=50, device='cuda'):
    """
    使用 PyTorch 对三维数据进行 TGV 去噪
    :param volume: 输入的三维数据（NumPy 数组）
    :param lambda_tgv: 正则化参数
    :param alpha0, alpha1: TGV 权重参数
    :param max_iter: 最大迭代次数
    :param device: 计算设备 ('cuda' 或 'cpu')
    :return: 去噪后的三维数据（NumPy 数组）
    """
    # 将数据转换为 PyTorch 张量并移动到指定设备
    volume_tensor = torch.from_numpy(volume).float().to(device)
    u = volume_tensor.clone()  # 去噪后的图像
    v = torch.zeros_like(u, device=device)  # 辅助变量

    for i in range(max_iter):
        print(f'Iteration {i + 1} of {max_iter}')

        # 计算梯度
        grad_u = torch.gradient(u, dim=(0, 1, 2))
        grad_v = torch.gradient(v, dim=(0, 1, 2))

        # 更新 u 和 v
        u = u - lambda_tgv * (u - volume_tensor) + alpha1 * (grad_u[0] - v) / (torch.abs(grad_u[0] - v) + 1e-10)
        v = v + alpha0 * grad_v[0] / (torch.abs(grad_v[0]) + 1e-10)

        # 投影到合理范围
        u = torch.clamp(u, 0, torch.max(volume_tensor))
        v = torch.clamp(v, -1, 1)

    # 将结果转换回 NumPy 数组
    return u.cpu().numpy()

def load_nii_gz(file_path):
    """
    加载 .nii.gz 文件并返回 NumPy 数组
    :param file_path: .nii.gz 文件路径
    :return: 三维数据（NumPy 数组）
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img.affine, img.header

def save_nii_gz(data, affine, header, output_path):
    """
    将 NumPy 数组保存为 .nii.gz 文件
    :param data: 三维数据（NumPy 数组）
    :param affine: 仿射矩阵
    :param header: 文件头信息
    :param output_path: 输出文件路径
    """
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, output_path)

if __name__ == '__main__':
    # 示例：读取 .nii.gz 文件并进行 TGV 去噪
    input_path = r'D:\Python_code\CBCT-to-CT-CycleGAN\test_data\predict.nii.gz'
    output_path = r'D:\Python_code\CBCT-to-CT-CycleGAN\test_data\predict-tgv-torch-improved.nii.gz'

    # 加载数据
    volume, affine, header = load_nii_gz(input_path)

    # 检查是否有 GPU 可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 对三维数据进行 TGV 去噪
    denoised_volume = tgv_denoise_3d_torch(
        volume,
        lambda_tgv=0.05,  # 调整 lambda_tgv
        alpha0=.5,       # 调整 alpha0
        alpha1=.8,       # 调整 alpha1
        max_iter=50,      # 增加迭代次数
        device=device
    )

    # 保存去噪后的数据
    save_nii_gz(denoised_volume, affine, header, output_path)
    print(f"去噪后的数据已保存到: {output_path}")