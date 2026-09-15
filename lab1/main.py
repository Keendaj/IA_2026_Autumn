import itertools

import numpy as np

MID_A1 = np.array([[0.95, 1.00],
                    [1.05, 1.00],
                    [1.10, 1.00]])

MID_A2 = np.array([[1.10, 0.90, 1.10],
                    [1.40, 1.00, 0.80],
                    [0.80, 1.40, 1.20]])


def check(name, mid, rad, delta, A):
    lo, hi = mid - delta * rad, mid + delta * rad
    inside = np.all((A >= lo - 1e-9) & (A <= hi + 1e-9))
    rank = np.linalg.matrix_rank(A, tol=1e-9)
    print(f"--- {name} ---")
    print(f"delta_min = {delta:.6f}")
    print("A' =\n", A)
    print(f"A' принадлежит интервальной матрице: {inside}")
    print(f"ранг(A') = {rank}\n")


def solve_rectangular(mid, second_col_exact):
    c = mid[:, 0]
    m2 = mid[0, 1]
    c_min, c_max = c.min(), c.max()
    lam = (c_max + c_min) / 2

    if second_col_exact:
        delta = (c_max - c_min) / 2
        rad = np.array([[1, 0], [1, 0], [1, 0]], dtype=float)
        col2 = np.full(3, m2)
    else:
        delta = (c_max - c_min) / (c_max + c_min + 2 * m2)
        rad = np.ones((3, 2))
        col2 = np.where(c == c_max, m2 + delta, np.where(c == c_min, m2 - delta, m2))

    A = np.column_stack([lam * col2, col2])
    return delta, rad, A


def solve_square(Ac):   
    xs = np.array([0.0, 1.0, -1.0, 2.0])
    best_delta, best_S = np.inf, None
    for bits in itertools.product([1, -1], repeat=9):
        S = np.array(bits, dtype=float).reshape(3, 3)
        ys = [np.linalg.det(Ac + x * S) for x in xs]
        for root in np.roots(np.polyfit(xs, ys, 3)):
            if abs(root.imag) < 1e-9 and 1e-9 < root.real < best_delta:
                best_delta, best_S = root.real, S
    A = Ac + best_delta * best_S
    return best_delta, np.ones((3, 3)), A


if __name__ == "__main__":
    delta, rad, A = solve_rectangular(MID_A1, second_col_exact=True)
    check("A1, регрессия", MID_A1, rad, delta, A)

    delta, rad, A = solve_rectangular(MID_A1, second_col_exact=False)
    check("A1, томография", MID_A1, rad, delta, A)

    delta, rad, A = solve_square(MID_A2)
    check("A2", MID_A2, rad, delta, A)
