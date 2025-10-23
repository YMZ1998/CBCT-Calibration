import os
import os.path

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error


def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def compare_images(mhd_file_path1, mhd_file_path2, mhd_file_path3, mhd_file_path4, title='', cmap='jet'):
    image1 = sitk.ReadImage(mhd_file_path1)
    image2 = sitk.ReadImage(mhd_file_path2)
    image3 = sitk.ReadImage(mhd_file_path3)
    image4 = sitk.ReadImage(mhd_file_path4)

    image_array1 = sitk.GetArrayFromImage(image1)
    image_array2 = sitk.GetArrayFromImage(image2)
    image_array3 = sitk.GetArrayFromImage(image3)
    image_array4 = sitk.GetArrayFromImage(image4)

    if image_array1.shape != image_array2.shape or image_array3.shape != image_array4.shape:
        print(image_array1.shape, image_array2.shape, image_array3.shape, image_array4.shape)
        raise ValueError("Image dimensions do not match for corresponding pairs.")

    # Normalize images
    normalized_image1 = normalize_image(image_array1)
    normalized_image2 = normalize_image(image_array2)
    normalized_image3 = normalize_image(image_array3)
    normalized_image4 = normalize_image(image_array4)

    # Difference calculation
    difference_array1 = normalized_image1 - normalized_image2
    difference_array2 = normalized_image3 - normalized_image4

    # Compute MSE and SSIM for both pairs
    mse1 = mean_squared_error(normalized_image1, normalized_image2)
    mse2 = mean_squared_error(normalized_image3, normalized_image4)

    ssim1 = ssim(normalized_image1, normalized_image2, data_range=1.0)
    ssim2 = ssim(normalized_image3, normalized_image4, data_range=1.0)

    print(f"Pair A MSE: {mse1}, SSIM: {ssim1}")
    print(f"Pair B MSE: {mse2}, SSIM: {ssim2}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plt.suptitle(title)

    # Image pair A (slice or 2D view)
    axes[0, 0].imshow(normalized_image1, cmap='gray')
    axes[0, 0].set_title(f"Image 1 - Pair A")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(normalized_image2, cmap='gray')
    axes[0, 1].set_title(f"Image 2 - Pair A")
    axes[0, 1].axis('off')

    axes[0, 2].set_title(f"Difference - Pair A")
    axes[0, 2].axis('off')
    fig.colorbar(axes[0, 2].imshow(difference_array1, cmap=cmap), ax=axes[0, 2])

    # Image pair B (slice or 2D view)
    axes[1, 0].imshow(normalized_image3, cmap='gray')
    axes[1, 0].set_title(f"Image 1 - Pair B")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(normalized_image4, cmap='gray')
    axes[1, 1].set_title(f"Image 2 - Pair B")
    axes[1, 1].axis('off')

    axes[1, 2].set_title(f"Difference - Pair B")
    axes[1, 2].axis('off')
    fig.colorbar(axes[1, 2].imshow(difference_array2, cmap=cmap), ax=axes[1, 2])

    plt.tight_layout()
    plt.show()


def test_case(case_id):
    src_path = r'D:\Data\cbct\DR0707\head'
    # src_path = r'D:\Data\cbct\DR0703\111'
    case=os.listdir(src_path)[case_id]
    mhd_file_path1 = os.path.join(src_path, case, 'a_mhd.mhd')
    mhd_file_path2 = os.path.join(src_path, case, 'drra.mhd')
    mhd_file_path3 = os.path.join(src_path, case, 'b_mhd.mhd')
    mhd_file_path4 = os.path.join(src_path, case, 'drrb.mhd')

    compare_images(mhd_file_path1, mhd_file_path2, mhd_file_path3, mhd_file_path4, title='Comparison for Case: ' + case)

    # plt.show()


if __name__ == '__main__':
    # for case in range(0,7,1):
    #     test_case(case)
    test_case(0)
    # test_case('45_2')
    # test_case('315')
    # test_case('315_1')
    # test_case("45")
    # test_case("45_xyz+1")
    # test_case("315")
    # test_case("315_xyz+1")
    # test_case("y+1")
    # test_case("y-1")
    # test_case("z+1")
    # test_case("z-1")
