import sys
import numpy as np

import img_util
import cubic_spline


def gauss_seidel(A, b, xi, err=1e-3, max_iter=10000):
    n = len(b)
    x_new = xi.copy()
    calc_err = err + 1
    iter_count = 0

    while calc_err > err and iter_count < max_iter:
        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            if A[i][i] == 0:
                raise ZeroDivisionError(f"A[{i}][{i}] is zero. Cannot divide.")
            x_new[i] = (b[i] - sigma) / A[i][i]

        delta_x = np.abs(x_new - xi)
        norm_delta_x = np.linalg.norm(delta_x)
        norm_xi = np.linalg.norm(xi)
        calc_err = norm_delta_x / (norm_xi + 1e-8)  # avoid division by zero
        xi = x_new.copy()
        iter_count += 1

    if iter_count == max_iter:
        print("Warning: Gauss-Seidel did not converge within max iterations.")

    return x_new


if __name__ == "__main__":
    sample_points = [(0, 0), (1, 3), (2, 1), (4, 5)]
    A, b = cubic_spline.get_a_matrix_and_b_vector(sample_points)

    if len(A) == 0 or len(b) == 0:
        sys.exit("Invalid system. Not enough points.")

    b = b.flatten()
    xi = np.ones_like(b)

    try:
        x = gauss_seidel(A, b, xi)
        print("Solution x:\n", x)
    except ZeroDivisionError as e:
        print("Error during solving:", e)

    img_util.plotPoints(sample_points, 'Sample Points')
