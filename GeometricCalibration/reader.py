"""This module contains some function for I/O purposes."""
import os
import numpy as np
from utils import angle2rotm, deg2rad


def read_bbs_ref_file(filename):
    """Read phantom reference file with bbs coordinates

    :param filename: path to file
    :type filename: str
    :return: Array containing bbs coordinates [x,y,z]
    :rtype: numpy.array
    """
    # Load reference 3D BBs
    # spiral distribution visible on x-y plane
    # [X,Y,Z] coordinates of Brandis phantom reference BB
    bbs = np.loadtxt(filename, delimiter=" ")

    # spiral distribution visible on x-z plane
    # rotation along x about 90
    T_map = angle2rotm(deg2rad(90), deg2rad(0), deg2rad(0))
    bbs = np.append(bbs, np.ones((bbs.shape[0], 1)), axis=1)  # homogeneous
    bbs = np.matmul(T_map, bbs.T).T
    bbs = bbs[:, 0:3]  # back to not homogeneous

    return bbs


def read_raw_image(filename, proj_size=[1420, 1420], dtype=np.uint16):
    import re

    width, height = proj_size
    match = re.search(r"([-+]?\d+\.\d+)\.raw", filename)
    angle = None
    if match:
        angle = float(match.group(1))

    file_size = width * height * np.dtype(dtype).itemsize
    with open(filename, 'rb') as f:
        raw_data = f.read(file_size)

    image = np.frombuffer(raw_data, dtype=dtype)
    image = image.reshape((height, width))

    print(f"读取 {filename} 完成, 角度: {angle}")

    return image, angle


def read_projection_file(proj_folder, proj_size):
    raw_files = sorted([f for f in os.listdir(proj_folder) if f.endswith(".raw")])[:100]
    is_a = False
    if "A" in raw_files[0]:
        is_a = True
    print(raw_files)
    proj_file = []
    angles = []
    for i, f in enumerate(raw_files):
        image, angle = read_raw_image(os.path.join(proj_folder, f), proj_size, np.uint16)
        if is_a:
            angle = angle + 45
        else:
            angle = angle - 45
        # angles = np.fmod(angles + 360.0, 360.0)  # Normalize angle to [0, 360)
        # angles = 360.0 - angles  # Flip the angle
        angle = np.deg2rad(angle)
        proj_file.append(os.path.join(proj_folder, f))
        angles.append(angle)

    return proj_file, angles


def read_img_label_file(filename):
    """Read imgLabels.txt file contained in .raw projection's directory.
    This File contains information about path and the gantry angle of every
    .raw projection.

    :param filename: path to file
    :type filename: str
    :return: list with path and list with angles for every row in
     imgLabels.txt file
    :rtype: list
    """
    # read image labels
    with open(filename, "r") as f:
        f.readline()  # Skip first row (header)

        file = f.readlines()

        proj_file = []  # last part of the projection file path
        angles = []  # angles of rotation of each image
        for line in file:
            proj_file.append(os.path.basename(line.split(" ")[0]))
            angles.append(float(line.split(" ")[1]))

    return proj_file, angles


def read_projection_raw(filename, dim):
    """Read .raw file and load it into a Numpy array.

    :param filename: path to file
    :type filename: str
    :param dim: Dimension of image
    :type dim: list
    :return: array containing loaded .raw image
    :rtype: numpy.array
    """
    image = np.fromfile(filename, dtype="uint16", sep="")
    image = np.reshape(image, newshape=[dim[1], dim[0]]).T
    return image


def read_projection_hnc(filename, dim):
    """Read .hnc file and load it into a Numpy array.

    :param filename: path to file
    :type filename: str
    :param dim: Dimension of image
    :type dim: list
    :return: array containing loaded .raw image
    :rtype: numpy.array
    """
    with open(filename, "rb") as f:
        # Read and discard header's bytes
        f.read(512)
        image = np.frombuffer(f.read(), dtype=np.uint16)

        # Change the shape of the array to the actual shape of the picture
        image.shape = (dim[0], dim[1])

    return image
