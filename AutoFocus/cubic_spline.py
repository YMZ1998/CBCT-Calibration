import sys
import numpy as np
import img_util


def get_a_matrix_and_b_vector(points):
    if len(points) <= 1:
        print('Not enough points!')
        return [], []

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    n = len(x) - 1

    A = np.zeros((4 * n, 4 * n))
    b = np.zeros((4 * n, 1))

    row = 0
    while row < 2 * n:
        i = row // 2
        A[row][4 * i + 0] = x[i] ** 3
        A[row][4 * i + 1] = x[i] ** 2
        A[row][4 * i + 2] = x[i]
        A[row][4 * i + 3] = 1
        b[row] = y[i]

        A[row + 1][4 * i + 0] = x[i + 1] ** 3
        A[row + 1][4 * i + 1] = x[i + 1] ** 2
        A[row + 1][4 * i + 2] = x[i + 1]
        A[row + 1][4 * i + 3] = 1
        b[row + 1] = y[i + 1]

        row += 2

    while row < 3 * n - 1:
        i = row - 2 * n
        xi = x[i + 1]
        A[row][4 * i + 0] = 3 * xi ** 2
        A[row][4 * i + 1] = 2 * xi
        A[row][4 * i + 2] = 1
        A[row][4 * (i + 1) + 0] = -3 * xi ** 2
        A[row][4 * (i + 1) + 1] = -2 * xi
        A[row][4 * (i + 1) + 2] = -1
        row += 1

    while row < 4 * n - 2:
        i = row - (3 * n - 1)
        xi = x[i + 1]
        A[row][4 * i + 0] = 6 * xi
        A[row][4 * i + 1] = 2
        A[row][4 * (i + 1) + 0] = -6 * xi
        A[row][4 * (i + 1) + 1] = -2
        row += 1

    A[row][0] = 6 * x[0]
    A[row][1] = 2
    A[row + 1][4 * n - 4] = 6 * x[n]
    A[row + 1][4 * n - 3] = 2

    return A, b


if __name__ == '__main__':
    sample_points = [(0, 0), (1, 3), (2, 1), (4, 5)]
    A, b = get_a_matrix_and_b_vector(sample_points)
    print("A matrix:\n", A)
    print("b vector:\n", b)
    img_util.plotPoints(sample_points, 'Points')
