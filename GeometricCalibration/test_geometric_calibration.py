from click.testing import CliRunner

import cli
from calibration import run_calibration


# def run_cli():
#     # cli.main()
#     runner = CliRunner()
#     # result = runner.invoke(cli.main, ["--help"])
#     # print(result.output)
#     input_dir = r"D:\Data\cbct\CBCT0707"
#     ref_file = r"./geometric_calibration/app_data/ref_brandis.txt"
#
#     print('start')
#     result = runner.invoke(cli.main, [
#         "--mode", "cbct",
#         "--input_path", str(input_dir),
#         "--ref", str(ref_file),
#         "--sad", "940",
#         "--sid", "1490"
#     ])
#     print('end')
#     print(result.output)
#     print(result.exit_code)
#     print(result.exception)
def vis_bbs():
    import numpy as np
    import matplotlib.pyplot as plt

    bbs = np.loadtxt(r"./app_data/ref_brandis.txt")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(bbs[:, 0], bbs[:, 1], bbs[:, 2], c='b', s=20)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Brandis Phantom - BB Coordinates')
    plt.show()


if __name__ == "__main__":
    vis_bbs()
    run_calibration(
        mode="cbct",
        input_path=r"D:\Data\cbct\CBCT0707",
        ref="./app_data/ref_brandis.txt",
        sad=940,
        sid=1450,
    )
