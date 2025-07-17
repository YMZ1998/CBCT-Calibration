import sys
import numpy as np
import cubic_spline


def solve(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    x = np.zeros((n, 1))

    # Elimination with partial pivoting
    for i in range(n - 1):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if A[max_row, i] == 0:
            raise ValueError(f"Matrix is singular at column {i}")

        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

        for row in range(i + 1, n):
            f = A[row, i] / A[i, i]
            A[row, i:] -= f * A[i, i:]
            b[row] -= f * b[i]

    # Back substitution
    for i in range(n - 1, -1, -1):
        s = np.dot(A[i, i + 1:], x[i + 1:])
        x[i] = (b[i] - s) / A[i, i]

    return x


if __name__ == '__main__':
    sample_points = [(0, 0), (1, 3), (2, 1), (4, 5)]
    A, b = cubic_spline.get_a_matrix_and_b_vector(sample_points)

    print("A matrix:\n", A)
    print("b vector:\n", b)

    try:
        x = solve(A, b)
        print("Solution x:\n", x)
    except ValueError as e:
        print("Error:", e)

    sys.exit()
