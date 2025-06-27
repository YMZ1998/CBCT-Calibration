from click.testing import CliRunner
from trimesh.path.packing import paths

from geometric_calibration import cli

if __name__ == "__main__":
    # cli.main()
    runner = CliRunner()
    # result = runner.invoke(cli.main, ["--help"])
    # print(result.output)
    input_dir = r"D:\Data\cbct\250613模体数据\A"
    ref_file = r"./geometric_calibration/app_data/ref_brandis.txt"

    print('start')
    result = runner.invoke(cli.main, [
        "--mode", "cbct",
        "--input_path", str(input_dir),
        "--ref", str(ref_file),
        "--sad", "1172.2",
        "--sid", "1672.2"
    ])
    print('end')
    print(result.output)
    print(result.exit_code)
    print(result.exception)
