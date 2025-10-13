from Scripts.suv_hist import plot_suv_histogram

if __name__ == "__main__":
    pet_files = [
        r"D:\Data\pet\SUV.nii.gz",
        r"D:\Data\pet\SUV2.nii.gz"
    ]
    for pet_file in pet_files:
        plot_suv_histogram(pet_file, roi_file=None, suv_threshold=2.5)
