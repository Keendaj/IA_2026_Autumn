from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class I:
    __array_ufunc__ = None

    def __init__(self, lo, hi=None): self.lo, self.hi = lo, lo if hi is None else hi
    def __add__(self, o): o = I.of(o); return I(self.lo + o.lo, self.hi + o.hi)
    def __neg__(self): return I(-self.hi, -self.lo)
    def __sub__(self, o): return self + -I.of(o)
    def __mul__(self, o): o = I.of(o); p = self.lo*o.lo, self.lo*o.hi, self.hi*o.lo, self.hi*o.hi; return I(min(p), max(p))
    def __pow__(self, n): lo, hi = sorted((self.lo**n, self.hi**n)); return I(0 if n % 2 == 0 and self.lo < 0 < self.hi else lo, hi)
    def __and__(self, o): return I(max(self.lo, o.lo), min(self.hi, o.hi))
    def __repr__(self): return f"[{self.lo:9.4f}, {self.hi:9.4f}]"
    __radd__, __rmul__ = __add__, __mul__
    of = staticmethod(lambda v: v if isinstance(v, I) else I(v))
    rad = property(lambda self: (self.hi - self.lo) / 2)
    mid = property(lambda self: (self.lo + self.hi) / 2)
    mag = property(lambda self: max(abs(self.lo), abs(self.hi)))


dist = lambda F, G: max(abs(F.lo - G.lo), abs(F.hi - G.hi))
hull = lambda F, G: I(min(F.lo, G.lo), max(F.hi, G.hi))
halves = lambda X: (I(X.lo, X.mid), I(X.mid, X.hi))
cmp = lambda p, q: "<=" if p <= q else "> "
out = lambda name: Path(__file__).with_name(name)
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
BLUE_LINES = LinearSegmentedColormap.from_list("blue_lines", ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def baumann(D, X):
    p = np.clip(D.mid / D.rad, -1, 1)
    return X.mid - p * X.rad, X.mid + p * X.rad


def analyze(n, tex, P, X, f, df, slope, forms):
    a, b = X.lo, X.hi
    mv = lambda X, c: f(c) + df(X) * (X - c)
    xs, cs = np.linspace(a, b, 1001), np.linspace(a, b, 5)
    crit = lambda p: np.array([a, b, *(r.real for r in np.roots(p) if abs(r.imag) < 1e-9 and a <= r.real <= b)])
    t = crit(np.polyder(P))
    ran = I(f(t).min(), f(t).max())
    L = np.polyval(np.polyder(np.abs(P)), X.mag)
    c_lo, c_hi = baumann(df(X), X)
    est = {"B.1 natural": f(X),
           **{k: F(X) for k, F in forms.items()},
           **{f"B.3 mv, c={c:g}": mv(X, c) for c in cs},
           **{f"B.4 slope, c={c:g}": f(c) + slope(X, c) * (X - c) for c in cs},
           "B.5 bicentered": mv(X, c_lo) & mv(X, c_hi)}

    print(f"=== f{n}(x) = {tex},  X = [{a:g}, {b:g}] ===")
    print(f"A.  ran(f, X) = {ran}")
    for k, F in est.items():
        print(f"{k:28}{F}  rad = {F.rad:8.4f}  dist = {dist(F, ran):8.4f}")
    r, R = est["B.1 natural"].rad, L * X.rad
    print(f"C.1 L = {L:g}")
    print(f"C.2 L*rad X = {R:.4f}")
    print(f"C.3 rad F(X) = {r:8.4f} {cmp(r, R)} L*rad X = {R:.4f}\n")

    fig, ax = plt.subplots()
    ax.plot(xs, f(xs), label=f"$f_{n}(x) = {tex}$")
    ax.hlines([ran.lo, ran.hi], a, b, colors="r", linestyles="--", label="ran$(f, X)$")
    ax.grid(); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    fig.savefig(out(f"f{n}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze2(n, tex, X1, X2, f, fx, fy, crit, forms, centers):
    D1, D2 = fx(X1, X2), fy(X1, X2)
    mv = lambda c: f(*c) + D1 * (X1 - c[0]) + D2 * (X2 - c[1])
    pts = [(x, y) for x in (X1.lo, X1.hi) for y in (X2.lo, X2.hi)]
    pts += [p for p in crit if X1.lo <= p[0] <= X1.hi and X2.lo <= p[1] <= X2.hi]
    vals = [f(*p) for p in pts]
    ran, F = I(min(vals), max(vals)), f(X1, X2)
    pt = lambda p: f"({p[0]:g}, {p[1]:g})"
    row = lambda k, G: print(f"{k:36}{G}  rad = {G.rad:8.4f}  dist = {dist(G, ran):8.4f}")

    print(f"=== f{n}(x, y) = {tex},  X = [{X1.lo:g}, {X1.hi:g}] x [{X2.lo:g}, {X2.hi:g}] ===")
    print(f"E.2 ran(f, X) = {ran}  min at {pt(pts[np.argmin(vals)])}, max at {pt(pts[np.argmax(vals)])};  "
          + "candidates: " + ", ".join(f"{pt(p)} -> {v:g}" for p, v in zip(pts, vals)))
    row("E.3 natural", F)
    for k, G in forms.items():
        row(k, G(X1, X2))
    for c, note in [((X1.mid, X2.mid), " = mid X"), *((c, "") for c in centers)]:
        row(f"E.5 mv, c={pt(c)}{note}", mv(c))
    (l1, u1), (l2, u2) = baumann(D1, X1), baumann(D2, X2)
    row("E.6 bicentered", mv((l1, l2)) & mv((u1, u2)))
    print(f"    c- = ({l1:.4f}, {l2:.4f}), c+ = ({u1:.4f}, {u2:.4f})")
    for k, parts in (("x", [(A, X2) for A in halves(X1)]), ("y", [(X1, B) for B in halves(X2)])):
        G1, G2 = (f(*p) for p in parts)
        row(f"E.7 split by {k}, hull", hull(G1, G2))
        print(f"    halves: {G1} and {G2}")
    L1, L2 = D1.mag, D2.mag
    R = L1 * X1.rad + L2 * X2.rad
    print(f"E.8 fx(X) = {D1}, fy(X) = {D2}  =>  L1 = {L1:g}, L2 = {L2:g}")
    print(f"    rad F(X) = {F.rad:8.4f} {cmp(F.rad, R)} L1*rad X1 + L2*rad X2 = {R:.4f}\n")

    xs, ys = np.meshgrid(np.linspace(X1.lo, X1.hi, 201), np.linspace(X2.lo, X2.hi, 201))
    zs = f(xs, ys)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(xs, ys, zs, cmap=BLUES, edgecolor="none")
    ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
    fig.colorbar(surf, shrink=0.6, pad=0.12, label=f"$z = f_{n}(x, y)$")
    fig.savefig(out(f"f{n}_surface.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    cs = ax.contour(xs, ys, zs, levels=np.arange(np.floor(ran.lo) + 0.5, ran.hi + 0.01, 0.5), cmap=BLUE_LINES, linewidths=1)
    ax.clabel(cs, levels=cs.levels[1::2], fmt="%g", fontsize=8, colors="#52514e")
    for v, color, name in ((ran.lo, "#eb6834", "min"), (ran.hi, "#1baf7a", "max")):
        p = np.array([q for q, w in zip(pts, vals) if w == v]).T
        ax.plot(*p, "o", ms=9, color=color, mec="white", mew=1.5, clip_on=False, zorder=3,
                label=f"{name} $f_{n}$ = {v:g}")
    ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.savefig(out(f"f{n}_levels.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


analyze(1, "x^3 - 3x^2 + 2", [1, -3, 0, 2], I(0.0, 3.0),
        f=lambda x: x**3 - 3*x**2 + 2,
        df=lambda x: 3*x**2 - 6*x,
        slope=lambda X, c: X**2 + c*X + c**2 - 3*(X + c),
        forms={"B.2 (x-1)((x-1)^2-3)": lambda X: (X - 1) * ((X - 1)**2 - 3)})

analyze(2, "x^5 - 4x^3 + 3x", [1, 0, -4, 0, 3, 0], I(-2.0, 2.0),
        f=lambda x: x**5 - 4*x**3 + 3*x,
        df=lambda x: 5*x**4 - 12*x**2 + 3,
        slope=lambda X, c: X**4 + c*X**3 + c**2*X**2 + c**3*X + c**4 - 4*(X**2 + c*X + c**2) + 3,
        forms={"B.2 x((x^2-2)^2-1)": lambda X: X * ((X**2 - 2)**2 - 1)})

X1, X2 = I(-1.0, 2.0), I(-1.0, 2.0)
analyze2(3, "x^2 + y^2 + xy - 2x - y", X1, X2,
         f=lambda x, y: x**2 + y**2 + x*y - 2*x - y,
         fx=lambda x, y: 2*x + y - 2,
         fy=lambda x, y: 2*y + x - 1,
         crit=[(1.0, 0.0), *((x, (1 - x) / 2) for x in (X1.lo, X1.hi)), *(((2 - y) / 2, y) for y in (X2.lo, X2.hi))],
         forms={"E.4 ((x-1)^2+y^2+(x+y-1)^2)/2-1": lambda x, y: 0.5*((x - 1)**2 + y**2 + (x + y - 1)**2) - 1},
         centers=[(1.0, 0.0)])
