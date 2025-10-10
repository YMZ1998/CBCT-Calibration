import itk
from itk import RTK as rtk

# 读取体积
volume = itk.imread("D:\\debug\\cbct\\A_output_1024_-0.67.mhd", itk.F)

# # 创建 GPU TV 去噪滤波器
# tv_filter = rtk.TotalVariationDenoisingBPDQImageFilter.New(volume)
# tv_filter.SetGamma(0.1)          # 梯度阈值
# # tv_filter.SetBeta(0.2)           # 迭代步长
# tv_filter.SetNumberOfIterations(10)
# tv_filter.Update()
#
# denoised_volume = tv_filter.GetOutput()

# 各向异性扩散去噪
diffusion = itk.GradientAnisotropicDiffusionImageFilter.New(volume)
diffusion.SetNumberOfIterations(10)
diffusion.SetTimeStep(0.01)
diffusion.SetConductanceParameter(0.1)
diffusion.Update()

denoised_volume = diffusion.GetOutput()


itk.imwrite(denoised_volume, "D:\\debug\\cbct\\volume_denoised_gpu.nii.gz")
