import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon
from skimage.transform import radon, resize


def shift_sinogram(sinogram, shift_val):
    sinogram_shifted = np.zeros_like(sinogram, dtype=np.float32)
    print("sinogram.shape:", sinogram.shape)
    sinogram_shifted[:180] = nd_shift(sinogram[:180], shift=shift_val, order=1, mode='nearest')
    sinogram_shifted[180:] = nd_shift(sinogram[180:], shift=-shift_val, order=1, mode='nearest')
    return sinogram_shifted

if __name__ == '__main__':
    # 1️⃣ 生成 phantom
    phantom = shepp_logan_phantom()
    # phantom = resize(phantom, (256, 256), mode='reflect', anti_aliasing=True)

    # 2️⃣ 生成 sinogram
    theta = np.linspace(0., 360., 360, endpoint=False)
    sinogram = radon(phantom, theta=theta, circle=True).T

    # 3️⃣ 偏移量
    shift_amounts = [0.5, -0.5, 9]
    sinograms_shifted = []

    for shift_val in shift_amounts:
        sinogram_shifted = shift_sinogram(sinogram, shift_val)
        sinograms_shifted.append(sinogram_shifted)

    fig, axes = plt.subplots(1, len(shift_amounts) + 1, figsize=(15, 4))

    axes[0].imshow(sinogram, cmap='gray', aspect='auto')
    axes[0].set_title("Original Sinogram")

    for i, shift_val in enumerate(shift_amounts):
        diff = sinograms_shifted[i] - sinogram  # 作差
        axes[i + 1].imshow(diff, cmap='bwr', aspect='auto', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        axes[i + 1].set_title(f"Difference ±{shift_val} px")

    plt.tight_layout()
    plt.show()

    reconstructions = []
    for sin_shifted in sinograms_shifted:
        recon = iradon(sin_shifted.T, theta=theta, filter_name='ramp', circle=True)
        reconstructions.append(recon)

    fig, axes = plt.subplots(1, len(shift_amounts) + 2, figsize=(15, 4))
    axes[0].imshow(phantom, cmap='gray')
    axes[0].set_title("Original Phantom")
    axes[0].axis('off')

    axes[1].imshow(iradon(sinogram.T, theta=theta, filter_name='ramp', circle=True), cmap='gray')
    axes[1].set_title("Reconstruction (No Shift)")
    axes[1].axis('off')

    for i, shift_val in enumerate(shift_amounts):
        axes[i + 2].imshow(reconstructions[i], cmap='gray')
        axes[i + 2].set_title(f"Reconstruction ±{shift_val} px")
        axes[i + 2].axis('off')

    plt.tight_layout()
    plt.show()
