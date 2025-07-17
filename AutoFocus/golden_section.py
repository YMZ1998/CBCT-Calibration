import math

import numpy as np

import cubic_spline
import gaussian_elimination
import img_util
import sharpness_calc

REQ_ERR = 0.01
GR = (math.sqrt(5) - 1) / 2


def eval_cubic_sharpness(x, coeffs, x_vals):
    left, right = 0, len(x_vals) - 1
    while left <= right:
        mid = (left + right) // 2
        if x_vals[mid - 1] <= x <= x_vals[mid]:
            a = coeffs[4 * (mid - 1)]
            b = coeffs[4 * (mid - 1) + 1]
            c = coeffs[4 * (mid - 1) + 2]
            d = coeffs[4 * (mid - 1) + 3]
            return a * x ** 3 + b * x ** 2 + c * x + d
        elif x < x_vals[mid - 1]:
            right = mid - 1
        else:
            left = mid + 1
    print(f"Value x={x} out of bounds ({x_vals[0]}, {x_vals[-1]})")
    return -1


def golden_section(xlow, xup, coeffs, x_vals):
    d = GR * abs(xup - xlow)
    x1, x2 = xlow + d, xup - d

    while True:
        d = GR * d
        y1 = eval_cubic_sharpness(x1, coeffs, x_vals)
        y2 = eval_cubic_sharpness(x2, coeffs, x_vals)

        if y1 >= y2:
            if max(abs(xup - x1), abs(x1 - x2)) < REQ_ERR:
                return (x1, y1)
            xlow = x2
        else:
            if max(abs(x1 - x2), abs(x2 - xlow)) < REQ_ERR:
                return (x2, y2)
            xup = x1

        x1 = xlow + d
        x2 = xup - d


if __name__ == '__main__':
    subject_name = '002'
    print(f"Processing subject: {subject_name}")
    points = sharpness_calc.get_points(subject_name, show_images=True)
    print(points)

    A, b = cubic_spline.get_a_matrix_and_b_vector(points)
    coeffs = gaussian_elimination.solve(A, b)
    x_vals = [p[0] for p in points]

    opt1 = golden_section(x_vals[0], (x_vals[0] + x_vals[-1]) / 2, coeffs, x_vals)
    opt2 = golden_section((x_vals[0] + x_vals[-1]) / 2, x_vals[-1], coeffs, x_vals)
    opt3 = golden_section(x_vals[0], x_vals[-1], coeffs, x_vals)

    opt = max([opt1, opt2, opt3], key=lambda o: o[1])
    print(f"Optimum found: X = {np.squeeze(opt[0]):.4f}, Y = {np.squeeze(opt[1]):.4f}")

    img_util.plotCubic(coeffs, x_vals, f'Cubic Spline for Subject #{subject_name}', opt[0], opt[1])
