import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize

# 1️⃣ 生成 phantom
phantom = resize(shepp_logan_phantom(), (256, 256), mode='reflect', anti_aliasing=True)

# 2️⃣ 生成 sinogram
theta = np.linspace(0., 180., 180, endpoint=False)
sinogram = radon(phantom, theta=theta, circle=True)

# 3️⃣ 反投影重建
reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp', circle=True)

# 4️⃣ 可视化
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(phantom, cmap='gray')
axes[0].set_title("Original Phantom")
axes[0].axis('off')

axes[1].imshow(sinogram, cmap='gray', aspect='auto')
axes[1].set_title("Sinogram")
axes[1].set_xlabel("Angle index")
axes[1].set_ylabel("Detector pixel")

axes[2].imshow(reconstruction_fbp, cmap='gray')
axes[2].set_title("Reconstruction (FBP)")
axes[2].axis('off')

plt.tight_layout()
plt.show()
