from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
"""
TFileHeader

TFileHeader 256
side: 1
mode: 1
frame_count: 2
reserved: [1]*252

TMemoryFrame 16400
Header: TFrameHeader(16)
Point: [TMemoryPoint(16)]*1024

TFrameHeader 16
StopPoint: 2
TimeStamp: 4
reserved: [2]*5

TMemoryPoint 16
Data: [2]*8

"""
DIR = Path(__file__).parent
file1 = DIR / "task_4/0.225_lvl_side_a_fast_data.bin"
file2 = DIR / "task_4/-0.205_lvl_side_a_fast_data.bin"

EPS = 1e-5
BLUE, INK, GRID = "#2a78d6", "#0b0b0b", "#e4e3df"

TFileHeaderType = np.dtype([
    ("side", "<u1"),
    ("mode", "<u1"),
    ("frame_count", "<u2"),
    ("reserved", "<u1", 252)
    ]
)

TFrameHeaderType = np.dtype([
    ("StopPoint", "u2"),
    ("TimeStamp", "u4"),
    ("reserved", "u2", 5),
])

TMemoryPointType = np.dtype([
    ("Data", "u2", 8)
])

TMemoryFrameType = np.dtype([
    ("Header", TFrameHeaderType),
    ("Point", TMemoryPointType, 1024),
])

def process_file(filepath: str):
    result = np.fromfile(filepath, dtype=TFileHeaderType, count=1)[0]
    frames = np.fromfile(filepath, dtype=TMemoryFrameType, offset=TFileHeaderType.itemsize)
    result = np.asarray(frames["Point"]["Data"], dtype=float)
    result = result / 16384 - 0.5
    result = result.reshape(100, 1024*8)

    return result

def calc_wid(mid, rad):
    lo = mid - rad
    hi = mid + rad
    
    inter_lo, inter_hi = lo.max(), hi.min()
    outer_lo, outer_hi = lo.min(), hi.max()

    inter_wid = inter_hi - inter_lo
    outer_wid = outer_hi - outer_lo

    return inter_wid, outer_wid

def calc_F(s, midX, radX, midY, radY, model="a"):
    if model == "a":
        midX = midX + s
    else:
        midX, radX = s * midX, abs(s) * radX

    mid = np.concatenate([np.ravel(midX), np.ravel(midY)])
    rad = np.concatenate([np.ravel(radX), np.ravel(radY)])
    
    inter_wid, outer_wid = calc_wid(mid, rad)

    return inter_wid / outer_wid

def calc_extremes(mid, rad):
    mid, rad = np.ravel(mid), np.ravel(rad)
    lo, hi = mid - rad, mid + rad
    idx = [lo.argmin(), lo.argmax(), hi.argmin(), hi.argmax()]

    return mid[idx], rad[idx]

def calc_pairs(midX, radX, midY, radY):
    pairs, counts = np.unique(np.stack([np.ravel(midX), np.ravel(radX), np.ravel(midY), np.ravel(radY)]),
                              axis=1, return_counts=True)

    return *np.ascontiguousarray(pairs), counts

def calc_Ji_vector(s, midX, radX, midY, radY, model="a"):
    if model == "a":
        midX = midX + s
    else:
        midX, radX = s * midX, abs(s) * radX

    inter_wid = np.minimum(midX + radX, midY + radY) - np.maximum(midX - radX, midY - radY)
    outer_wid = np.maximum(midX + radX, midY + radY) - np.minimum(midX - radX, midY - radY)

    return inter_wid / outer_wid

def calc_hull(mid, rad):
    hull_lo, hull_hi = np.min(mid - rad), np.max(mid + rad)

    return (hull_lo + hull_hi) / 2, (hull_hi - hull_lo) / 2

def calc_mode(mid, rad):
    lo, hi = np.ravel(mid - rad), np.ravel(mid + rad)
    ends = np.unique(np.concatenate([lo, hi]))
    count = np.searchsorted(np.sort(lo), ends[:-1], "right") - np.searchsorted(np.sort(hi), ends[:-1], "right")
    j = np.flatnonzero(count == count.max())
    mode_lo, mode_hi = ends[j[0]], ends[j[-1] + 1]

    return (mode_lo + mode_hi) / 2, (mode_hi - mode_lo) / 2

def calc_med_K(mid, rad):
    med_lo, med_hi = np.median(mid - rad), np.median(mid + rad)

    return (med_lo + med_hi) / 2, (med_hi - med_lo) / 2

def calc_med_P(mid, rad):
    return np.median(mid), np.median(rad)

def calc_grid(midX, radX, midY, radY, model):
    loX, hiX = np.min(midX - radX), np.max(midX + radX)
    loY, hiY = np.min(midY - radY), np.max(midY + radY)
    if model == "a":
        s_lo, s_hi = loY - hiX, hiY - loX
    else:
        ratios = np.outer([loY, hiY], 1 / np.array([loX, hiX]))
        s_lo, s_hi = ratios.min(), ratios.max()
    pad = (s_hi - s_lo) / 4

    return np.arange(s_lo - pad, s_hi + pad, min(EPS, (s_hi - s_lo) / 2000))

def calc_estimates(s, F):
    s_arg = s[F >= F.max() - 1e-12]
    s_cons = s[F >= 0]
    s_hat = (s_arg[0] + s_arg[-1]) / 2
    cons = (s_cons[0], s_cons[-1]) if s_cons.size else None

    return s_hat, (s_arg[0], s_arg[-1]), cons

def calc_refined(f, s, F):
    s_all, F_all = s, F
    while s[1] - s[0] > EPS:
        k = F.argmax()
        s = np.linspace(s[max(k - 1, 0)], s[min(k + 1, len(s) - 1)], 41)
        F = np.array([f(si) for si in s])
        s_all, F_all = np.concatenate([s_all, s]), np.concatenate([F_all, F])
    order = np.argsort(s_all)

    return s_all[order], F_all[order]

def print_estimates(model, case, s, F, with_cons=True):
    s_hat, arg, cons = calc_estimates(s, F)
    cons = cons if with_cons else None
    cons_str = f"[{cons[0]:.6f}, {cons[1]:.6f}]" if cons else ("empty" if with_cons else "-")
    print(f"{model}  {case:9}  s_hat = {s_hat:.6f},  Arg max F = [{arg[0]:.6f}, {arg[1]:.6f}],  "
          f"F >= 0: {cons_str},  F_max = {F.max():.6f}")

    return s, F, s_hat, arg, cons

def plot_F(model, results, cases, filename):
    nrows = (len(cases) + 1) // 2
    fig, axs = plt.subplots(nrows, 2, figsize=(12, 4.25 * nrows))
    for ax, case in zip(axs.flat, cases):
        s, F, s_hat, arg, cons = results[case]
        if cons:
            ax.axvspan(*cons, color=BLUE, alpha=0.12, lw=0, label=f"$F \\geq 0$: [{cons[0]:.6f}, {cons[1]:.6f}]")
        ax.plot(s, F, color=BLUE, lw=1.5)
        if arg[1] > arg[0]:
            ax.hlines(F.max(), *arg, color=INK, lw=3, label=f"Arg max $F$: [{arg[0]:.6f}, {arg[1]:.6f}]")
        ax.plot(s_hat, F.max(), "o", ms=8, color=INK, mec="white", mew=1.5, zorder=3,
                label=f"$\\hat{{{model}}}$ = {s_hat:.6f},  $F_{{max}}$ = {F.max():.4f}")
        ax.set(xlabel=f"${model}$", ylabel=f"$F({model})$", title=f"{case.split()[0]}: {TITLES[case]}")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(color=GRID, lw=0.8)
        ax.legend(loc="best", fontsize=8)
    for ax in axs.flat[len(cases):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

CASES = {
    "B.1": ("$Ji(s, X, Y)$", calc_extremes),
    "B.1 hull": ("$Ji(s, hull\\, X, hull\\, Y)$", calc_hull),
    "B.2": ("$Ji(s, mode\\, X, mode\\, Y)$", calc_mode),
    "B.3": ("$Ji(s, med_K X, med_K Y)$", calc_med_K),
    "B.4": ("$Ji(s, med_P X, med_P Y)$", calc_med_P),
}

AGGREGATES = {
    "B.1 min": ("$\\min_i\\, Ji(s, x_i, y_i)$", lambda Ji, counts: Ji.min()),
    "B.1 max": ("$\\max_i\\, Ji(s, x_i, y_i)$", lambda Ji, counts: Ji.max()),
    "B.1 mean": ("$mean_i\\, Ji(s, x_i, y_i)$", lambda Ji, counts: np.average(Ji, weights=counts)),
    "B.1 L1": ("$\\|Ji(s, x_i, y_i)\\|_1$", lambda Ji, counts: np.sum(counts * np.abs(Ji))),
    "B.1 L2": ("$\\|Ji(s, x_i, y_i)\\|_2$", lambda Ji, counts: np.sqrt(np.sum(counts * Ji**2))),
}

TITLES = {case: title for case, (title, _) in (CASES | AGGREGATES).items()}

if __name__ == "__main__":
    midX = process_file(file1)
    midY = process_file(file2)
    rad = 2**-14
    radX = np.full((100, 1024*8), rad)
    radY = np.full((100, 1024*8), rad)

    reps = {case: (reduce(midX, radX), reduce(midY, radY)) for case, (_, reduce) in CASES.items()}
    for case in ("B.2", "B.3", "B.4"):
        (mX, rX), (mY, rY) = reps[case]
        print(f"{case}: X -> [{mX - rX:.8f}, {mX + rX:.8f}],  Y -> [{mY - rY:.8f}, {mY + rY:.8f}]")

    all_pairs = {"a": calc_pairs(np.zeros_like(midX), radX, midY - midX, radY), "t": calc_pairs(midX, radX, midY, radY)}

    for model in ("a", "t"):
        *pairs, counts = all_pairs[model]
        results = {}
        for case, (X, Y) in reps.items():
            s = calc_grid(*X, *Y, model)
            F = np.array([calc_F(si, *X, *Y, model) for si in s])
            results[case] = print_estimates(model, case, s, F)

        s = calc_grid(*reps["B.1 hull"][0], *reps["B.1 hull"][1], model)
        s = np.linspace(s[0], s[-1], 501)
        if model == "a":
            s = np.union1d(s, pairs[2])
        F_all = np.array([[agg(calc_Ji_vector(si, *pairs, model), counts) for _, agg in AGGREGATES.values()]
                          for si in s])
        for j, (case, (_, agg)) in enumerate(AGGREGATES.items()):
            f = lambda si: agg(calc_Ji_vector(si, *pairs, model), counts)
            results[case] = print_estimates(model, case, *calc_refined(f, s, F_all[:, j]),
                                            case not in ("B.1 L1", "B.1 L2"))

        plot_F(model, results, ["B.1", "B.2", "B.3", "B.4"], f"F_{model}.png")
        plot_F(model, results, ["B.1", "B.1 hull", *AGGREGATES], f"B1_{model}.png")