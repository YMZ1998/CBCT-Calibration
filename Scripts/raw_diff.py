import os

from reg_diff_mhd import compare_images


def test_case():
    case1 = '202507011951'
    case2 = '202507011956'
    # src_path = r'D:\Data\cbct\move_after'
    src_path = r'D:\Data\cbct\DR0701\111'
    mhd_file_path1 = os.path.join(src_path, case1, 'a_mhd.mhd')
    mhd_file_path2 = os.path.join(src_path, case2, 'a_mhd.mhd')
    mhd_file_path3 = os.path.join(src_path, case1, 'b_mhd.mhd')
    mhd_file_path4 = os.path.join(src_path, case2, 'b_mhd.mhd')

    compare_images(mhd_file_path1, mhd_file_path2, mhd_file_path3, mhd_file_path4, title='Comparison for Case',
                   cmap='gray')


def test_case2():
    case1 = '202507011951'
    case2 = '202507011956'
    # src_path = r'D:\Data\cbct\move_after'
    src_path = r'D:\Data\cbct\DR0701\111'
    mhd_file_path1 = os.path.join(src_path, case1, 'drra.mhd')
    mhd_file_path2 = os.path.join(src_path, case2, 'drra.mhd')
    mhd_file_path3 = os.path.join(src_path, case1, 'drrb.mhd')
    mhd_file_path4 = os.path.join(src_path, case2, 'drrb.mhd')

    compare_images(mhd_file_path1, mhd_file_path2, mhd_file_path3, mhd_file_path4, title='Comparison for Case',
                   cmap='gray')


if __name__ == '__main__':
    test_case()
    # test_case2()
