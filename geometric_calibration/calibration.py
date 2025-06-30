"""Console script for geometric_calibration."""
import sys
import os
import click
import click_config_file

from geometric_calibration.cli import save_cli, REF_BBS_DEFAULT_PATH
from geometric_calibration.reader import read_bbs_ref_file
from geometric_calibration.geometric_calibration import (
    calibrate_cbct,
    calibrate_2d,
    save_lut,
    plot_calibration_results,
)


def run_calibration(
    mode="cbct",
    input_path=os.getcwd(),
    sad=940,
    sid=1490,
    ref=REF_BBS_DEFAULT_PATH,
):
    """Main entry point without CLI. Call this from other scripts."""

    bbs = read_bbs_ref_file(ref)

    print("Calibration Parameters:")
    print(f"Mode: '{mode}'")
    print(f"Input Path: '{input_path}'")
    print(f"SAD: '{sad}'")
    print(f"SID: '{sid}'\n")

    if mode == "cbct":
        print("Calibrating CBCT system. Please wait...")
        calibration_results = calibrate_cbct(input_path, bbs, sad, sid)
    elif mode == "2d":
        print("Calibrating 2D system. Please wait...")
        calibration_results = calibrate_2d(input_path, bbs, sad, sid)
    else:
        print(f"Mode '{mode}' not recognized.")
        return

    print("Saving...")

    save_flag = False
    while True:
        user_choice = input(
            "\nChoose an option:\n"
            "  s\tSave LUT\n"
            "  p\tPlot calibration result\n"
            "  c\tClose\n"
            "Your choice: "
        ).strip().lower()

        if user_choice == "s":
            save_cli(input_path, calibration_results, mode)
            save_flag = True
        elif user_choice == "p":
            plot_calibration_results(calibration_results)
        elif user_choice == "c":
            if not save_flag:
                confirm = input("New LUT not saved. Save now? [Y/n]: ").strip().lower()
                if confirm in ["", "y", "yes"]:
                    save_cli(input_path, calibration_results, mode)
            break
        else:
            print(f"Command '{user_choice}' not recognized.")
