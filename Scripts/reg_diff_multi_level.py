import os

from matplotlib import pyplot as plt

from reg_diff_mhd import compare_images


def test_level(case):
    # src_path = r'D:\Data\reg2d3d'
    # src_path = r'D:\Data\cbct\move_after'
    src_path = r'D:\Data\cbct\move_before'
    mhd_file_path1 = os.path.join(src_path, case, 'a_mhd.mhd')
    mhd_file_path3 = os.path.join(src_path, case, 'b_mhd.mhd')

    for i in range(2):
        mhd_file_path2 = os.path.join(src_path, case, 'level_' + str(i) + 'a.mhd')
        mhd_file_path4 = os.path.join(src_path, case, 'level_' + str(i) + 'b.mhd')

        compare_images(mhd_file_path1, mhd_file_path2, 'A ' + str(i))
        # compare_images(mhd_file_path3, mhd_file_path4, 'B ' + str(i))

    plt.show()


if __name__ == '__main__':
    test_level('DR1')
    # test_case('DR2')
