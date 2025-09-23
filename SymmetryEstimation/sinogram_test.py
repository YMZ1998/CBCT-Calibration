import matplotlib.pyplot as plt
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon

from SymmetryEstimation.sinogram import estimate_rotation_center
from SymmetryEstimation.sinogram_shift import shift_sinogram

if __name__ == '__main__':
    phantom = shepp_logan_phantom()
    # phantom = resize(phantom, (256, 256), mode='reflect', anti_aliasing=True)

    theta = np.linspace(0., 360., 360, endpoint=False)
    print(f"theta: {theta}°")

    sinogram = radon(phantom, theta=theta, circle=True).T

    sinogram = shift_sinogram(sinogram, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_title("Shepp-Logan Phantom")
    ax1.imshow(phantom, cmap='gray')
    ax1.axis('off')

    ax2.set_title("Simulated Sinogram")
    ax2.imshow(sinogram, cmap='gray', aspect='auto',
               extent=(0, sinogram.shape[1], theta[0], theta[-1]))
    ax2.set_xlabel("Detector pixel (u)")
    ax2.set_ylabel("Projection angle (°)")

    plt.tight_layout()
    plt.show()

    center_px, center_shift = estimate_rotation_center(sinogram)
    print(f"center_px : {center_px:.3f}")
    print(f"center_shift : {center_shift:.3f}")
