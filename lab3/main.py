from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

out = lambda name: Path(__file__).with_name(name)
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
COLORS = {"b": "#2a78d6", "A": "#eb6834", "Ab": "#1baf7a"}
STYLES = {"b": "-", "A": "--", "Ab": ":"}
TITLES = {"b": "b-коррекция", "A": "A-коррекция", "Ab": "Ab-коррекция"}
INK = "#0b0b0b"

# variant 1: A_k, b_k are the first k + 1 rows
A_LO = np.array([[0.65, 0.70], [0.75, 0.76], [0.80, 0.67], [-0.30, 0.82]])
A_HI = np.array([[1.25, 1.30], [1.35, 1.32], [1.40, 1.27], [0.30, 1.38]])
B_LO = np.array([2.75, 2.85, 2.90, 1.80])
B_HI = np.array([3.15, 3.25, 3.30, 2.20])
K = 0.8  # correction constant, K >= |T| for all three systems


def tol_rows(x, Alo, Ahi, blo, bhi):
    # generators of Tol (1.96): rad b_i - |mid b_i - a_i x| = min(lower(a_i x) - lower(b_i), upper(b_i) - upper(a_i x))
    P, Q = x[..., None, :] * Alo, x[..., None, :] * Ahi
    return np.minimum(np.minimum(P, Q).sum(-1) - blo, bhi - np.maximum(P, Q).sum(-1))


def vertices(G, h):
    # vertices of the polyhedron G z <= h: solutions of its square subsystems that satisfy every inequality;
    # all polyhedra below have rank G = dim z, so such a polyhedron is empty iff it has no vertices
    pts = []
    for idx in map(list, combinations(range(len(G)), G.shape[1])):
        if abs(np.linalg.det(G[idx])) > 1e-12:
            z = np.linalg.solve(G[idx], h[idx])
            if np.all(G @ z <= h + 1e-9):
                pts.append(z)
    pts = np.array(pts).reshape(-1, G.shape[1])
    return pts[np.unique(pts.round(9), axis=0, return_index=True)[1]]


def polygon(P):
    c = P.mean(0)
    return P[np.argsort(np.arctan2(*(P - c).T[::-1]))]


def area(P):
    x, y = polygon(P).T
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def sharaya(Alo, Ahi, blo, bhi):
    # system (1.88): lower(b_i) <= a x <= upper(b_i) for all a in vert A_i, as pairs of rows of G x <= h
    G, h = [], []
    for i in range(len(blo)):
        for a in product(*zip(Alo[i], Ahi[i])):
            G += [a, np.negative(a)]
            h += [bhi[i], -blo[i]]
    return np.array(G), np.array(h)


def lp_max(G, h, c):
    # max t subject to G z + c t <= h, solved by enumerating vertices
    pts = vertices(np.hstack([G, c[:, None]]), h)
    T = pts[:, -1].max()
    return T, pts[pts[:, -1] > T - 1e-9, :-1]


def tol_max(S):
    # Tol is the lower envelope of the linear functions upper(b_i) - a x, a x - lower(b_i), a in vert A_i (p. 135),
    # so max Tol is the LP  t -> max,  G x + t <= h  with G, h of (1.88)
    G, h = sharaya(*S)
    return lp_max(G, h, np.ones(len(G)))


chi = lambda lo, hi: lo / hi if abs(lo) <= abs(hi) else hi / lo  # Rachek functional (1.23)
fmt = lambda v: "(" + ", ".join(f"{round(x, 9) + 0:.6f}" for x in v) + ")"
iv = lambda lo, hi: "[" + ", ".join(f"[{l:.4f}, {u:.4f}]" for l, u in zip(np.ravel(lo), np.ravel(hi))) + "]"


def b_corr(S, Kb):
    # uniform widening b + K e, e = ([-1, 1], ..., [-1, 1]) (p. 138)
    return S[0], S[1], S[2] - Kb, S[3] + Kb


def a_corr(S, e):
    # A (-) E with E = ([-e, e]) by (2.8): [lower(a) + e, upper(a) - e], mid A is kept, rad A decreases by e (p. 143)
    return S[0] + e, S[1] - e, S[2], S[3]


def theorems(S):
    # the theorems are applied in turn until one of them answers whether Xi_tol is empty:
    # Thm 1.4.7 is only sufficient for emptiness, Thm 1.4.5 (Sharaya) always gives the answer
    Alo, Ahi, blo, bhi = S
    m = len(blo)
    c_a = [max(chi(l, u) for l, u in zip(Alo[i], Ahi[i]) if (l, u) != (0, 0)) for i in range(m)]
    c_b = [chi(l, u) for l, u in zip(blo, bhi)]
    zero = all(not l <= 0 <= u for l, u in zip(blo, bhi))
    holds = zero and all(p < q for p, q in zip(c_a, c_b))
    print(f"  Thm 1.4.7: 0 not in b_i: {zero};  max chi(a_ij) vs chi(b_i): "
          + ", ".join(f"{p:.4f} {'<' if p < q else '>='} {q:.4f}" for p, q in zip(c_a, c_b))
          + ("  =>  Xi_tol is empty" if holds else "  =>  conditions fail, no conclusion"))
    if holds:
        return np.empty((0, 2))
    G, h = sharaya(*S)
    V = vertices(G, h)
    if not len(V):
        print(f"  Thm 1.4.5: (1.88) has {len(G) // 2} double inequalities and no solution  =>  Xi_tol is empty")
        return V
    V = polygon(V)
    print(f"  Thm 1.4.5: (1.88) has {len(G) // 2} double inequalities, its solutions form a polygon with vertices "
          + ", ".join(fmt(v) for v in V) + "  =>  Xi_tol is non-empty")
    return V


def tol_report(S):
    T, taus = tol_max(S)
    tau, rows = taus[0], tol_rows(taus[0], *S)
    print(f"  max Tol = {T:.6f}")
    print(f"  A.1 Arg max Tol = {fmt(tau)}" + ("" if len(taus) == 1 else f";  Arg max is the hull of {taus.tolist()}"))
    print("      Tol_i(tau) = " + ", ".join(f"{v:.6f}" + "*" * bool(np.isclose(v, T)) for v in rows)
          + "   (* generators where the minimum is attained)")
    return T, tau


def distortion(S, C):
    # Hausdorff distance (1.25) between the original and corrected elements and the change of their radii;
    # mid is kept by every correction, so dist = |rad' - rad|
    ra, rb = (S[1] - S[0]) / 2, (S[3] - S[2]) / 2
    ra2, rb2 = (C[1] - C[0]) / 2, (C[3] - C[2]) / 2
    q = lambda r, r2: np.max(np.maximum(r2 / r, r / r2))
    return np.abs(rb2 - rb).max(), np.abs(ra2 - ra).max(), q(rb, rb2), q(ra, ra2)


def corrections(S, T, tau, Kc, verbose):
    s = np.abs(tau).sum()
    rad_a = (S[1] - S[0]) / 2
    e = Kc / s
    # Ab: the matrix gives K / 2 (not enough for solvability), then b is widened up to max Tol = T + K
    SA = a_corr(S, e / 2)
    TA = tol_max(SA)[0]
    Kb = T + Kc - TA
    if verbose:
        Delta = min((np.abs(tau) * rad_a[i]).sum() for i in range(len(S[2])))
        print(f"  (1.112): {-T / s:.6f} = |T| / (|tau_1| + |tau_2|) <= e <= min rad a_ij = {rad_a.min():.2f};  "
              f"Delta (1.113) = {Delta:.6f};  K = {Kc}:  e = K / (|tau_1| + |tau_2|) = {e:.6f}")
        print(f"  Ab: e = {e / 2:.6f},  max Tol(A (-) E, b) = {TA:.6f} < 0,  then b + K_b e with K_b = {Kb:.6f}")
    return {"b": b_corr(S, Kc), "A": a_corr(S, e), "Ab": b_corr(SA, Kb)}


def plot_corrected(k, corr, res):
    # both plots show the neighbourhood of the tolerance sets: the union of their bounding boxes with a margin
    V = np.vstack([r[2] for r in res.values()])
    (a0, b0), (a1, b1) = V.min(0), V.max(0)
    pad = 0.12 * max(a1 - a0, b1 - b0)
    xs, ys = np.meshgrid(np.linspace(a0 - pad, a1 + pad, 241), np.linspace(b0 - pad, b1 + pad, 241))
    zs = {n: tol_rows(np.stack([xs, ys], -1), *C).min(-1) for n, C in corr.items()}
    lo, hi = min(z.min() for z in zs.values()), max(z.max() for z in zs.values())
    fig = plt.figure(figsize=(15, 4.8))
    for j, (n, z) in enumerate(zs.items()):
        T, tau = res[n][:2]
        ax = fig.add_subplot(1, 3, j + 1, projection="3d", computed_zorder=False)
        surf = ax.plot_surface(xs, ys, z, cmap=BLUES, vmin=lo, vmax=hi, edgecolor="none", zorder=1)
        ax.contour(xs, ys, z, levels=[0], colors=COLORS[n], linewidths=2, linestyles=STYLES[n], zorder=4)
        ax.scatter(*tau, T, s=50, color=INK, edgecolor="white", depthshade=False, zorder=5)
        ax.set(xlabel="$x_1$", ylabel="$x_2$", zlim=(lo, hi), title=f"({'абв'[j]}) {TITLES[n]}: max Tol = {T:.3f}")
    fig.colorbar(surf, ax=fig.axes, shrink=0.6, pad=0.04, label="Tol$(x)$")
    fig.savefig(out(f"tol{k}_corr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 11))
    for j, (n, ax) in enumerate(zip([*corr, "all"], axs.flat)):
        for m in (corr if n == "all" else [n]):
            P = polygon(res[m][2])
            ax.fill(*P.T, color=COLORS[m], alpha=0.12 if n == "all" else 0.3, lw=0)
            ax.fill(*P.T, fill=False, ec=COLORS[m], lw=2, ls=STYLES[m], label=f"$\\Xi_{{tol}}$, {TITLES[m]}")
            if n != "all":
                ax.plot(*P.T, "o", ms=4, color=COLORS[m])
        T, tau = res["b" if n == "all" else n][:2]
        ax.plot(*tau, "o", ms=9, color=INK, mec="white", mew=1.5, zorder=3,
                label=f"Arg max Tol = ({tau[0]:.3f}, {tau[1]:.3f})")
        ax.set(xlim=(a0 - pad, a1 + pad), ylim=(b0 - pad, b1 + pad), xlabel="$x_1$", ylabel="$x_2$", aspect="equal",
               title=f"({'абвг'[j]}) " + ("сравнение" if n == "all" else TITLES[n]))
        ax.grid(color="#e4e3df", lw=0.8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(out(f"xi{k}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def solve(k, S):
    print(f"=================== A{k} x = b{k},  {len(S[2])}x2 ===================")
    print("A.  check of Xi_tol by the theorems until the first conclusive one")
    theorems(S)
    T, tau = tol_report(S)
    print("B.  corrections")
    corr = corrections(S, T, tau, K, True)
    res = {}
    for n, C in corr.items():
        print(f"--- {n}-correction: A = {iv(C[0], C[1])}")
        print(f"    b = {iv(C[2], C[3])}")
        V = theorems(C)
        Tc, tc = tol_report(C)
        db, da, qb, qa = distortion(S, C)
        print(f"      area of Xi_tol = {area(V):.6f};  dist(b, b') = {db:.6f}, dist(A, A') = {da:.6f}, "
              f"rad factor b: {qb:.4f}, A: {qa:.4f}")
        res[n] = Tc, tc, V
    print(f"C.  minimal corrections (K = |T| = {-T:.6f}):")
    for n, C in corrections(S, T, tau, -T, False).items():
        db, da, qb, qa = distortion(S, C)
        print(f"      {n:3} max Tol = {round(tol_max(C)[0], 9) + 0:.6f};  dist(b, b') = {db:.6f}, dist(A, A') = {da:.6f}, "
              f"rad factor b: {qb:.4f}, A: {qa:.4f}")
    plot_corrected(k, corr, res)
    print()


for k in (1, 2, 3):
    solve(k, (A_LO[:k + 1], A_HI[:k + 1], B_LO[:k + 1], B_HI[:k + 1]))
