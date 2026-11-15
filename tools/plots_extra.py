"""plots_extra.py: the report figures that have no counterpart in the paper, drawn in the paper's style
(the replication package's matplotlib settings, palette and helpers from ref/code/make_figures.py):
  grid_slices.pdf   bond price, default intensity and drift at four endowment levels on the paper grid and the
                    dense grids, and the ergodic density of the spread
  fp32_maps.pdf     |fp32 - fp64| of the price and the default intensity over the state space, paper grid and
                    2000 x 121 grid, with the zero-drift locus
  paper_figs        the paper's own figures, drawn from the package's stored solution (for side-by-side checks)
Usage: python tools/plots_extra.py <figure_dir>"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import plots  # noqa: E402  (imports make_figures with the package settings)
mf = plots.mf
plt = mf.plt
RES = os.path.join(ROOT, "results")


def load(name):
    L = lambda f: np.load(os.path.join(RES, name, f + ".npy"))
    return {k: L(k) for k in ("B", "z", "V", "q", "S", "c", "d", "g")}


def par():
    return mf.load_ou()[1]


def spread_density_line(ax, a, p, k, label):
    from scipy.ndimage import gaussian_filter1d
    coupon, base = p["delta"] + p["lam"], p["delta"] + p["r"]
    w = a["g"] / a["g"].sum()
    sp = 100 * (coupon / a["q"] - base)
    edges = np.linspace(0.0, 20.0, 401)
    h = np.histogram(sp.ravel(), bins=edges, weights=w.ravel())[0] / (edges[1] - edges[0])
    mf.line(ax, 0.5 * (edges[1:] + edges[:-1]), gaussian_filter1d(h, 4.0, mode="nearest"), k, label)


def grid_slices(figdir):
    """Price, default intensity and drift at -1 and +1 s.d. on three grids, and the spread density."""
    p = par()
    sols = [("p_cpu64", "$201\\times75$"), ("d500_cpu64", "$500\\times121$"), ("d2000_cpu64", "$2000\\times121$")]
    fig, axes = plt.subplots(2, 3, figsize=(mf.W, 0.6 * mf.W))
    for k, (name, lab) in enumerate(sols):
        a = load(name)
        B, z = a["B"][:-1], a["z"]
        for row, sdk in enumerate((-1, 1)):
            j = int(np.argmin(np.abs(z - sdk * p["sd"])))
            mf.line(axes[row, 0], B, a["q"][:-1, j], k, lab)
            mf.line(axes[row, 1], *mf.with_break(B, a["d"][:-1, j]), k, lab)
            if row == 0:
                mf.line(axes[0, 2], B, a["S"][:-1, j], k, lab)
        spread_density_line(axes[1, 2], a, p, k, lab)
    mf.zero_line(axes[0, 2])
    axes[1, 2].set_yscale("log"); axes[1, 2].set_ylim(1e-4, 2.0); axes[1, 2].set_xlim(0.0, 17.0)
    titles = ["bond price $q$, $-1$ s.d.", "default intensity $d$, $-1$ s.d.", "drift of debt $S$, $-1$ s.d.",
              "bond price $q$, $+1$ s.d.", "default intensity $d$, $+1$ s.d.", "ergodic density of the spread"]
    for ax, t in zip(axes.ravel(), titles):
        ax.set_title(t)
    for ax in axes[1, :2]:
        ax.set_xlabel("debt $B$")
    axes[0, 2].set_xlabel("debt $B$")
    axes[1, 2].set_xlabel("spread, percent")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, handlelength=2.4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=1.2, h_pad=1.0, rect=(0, 0.04, 1, 1))
    mf.FIG = figdir
    mf.save(fig, "grid_slices")


def fp32_maps(figdir):
    """log10 |fp32 - fp64| of q and d (GPU fp32 against CPU fp64), paper grid and 2000 x 121."""
    from matplotlib.colors import LogNorm
    pairs = [("p_gpu32", "p_cpu64", "$201\\times75$"), ("d2000_gpu32", "d2000_cpu64", "$2000\\times121$")]
    fig, axes = plt.subplots(2, 2, figsize=(mf.W, 0.62 * mf.W), sharey=True)
    for r, (a32, a64, lab) in enumerate(pairs):
        A, Bs = load(a32), load(a64)
        BB, ZZ = np.meshgrid(Bs["B"], Bs["z"], indexing="ij")
        for c, (f, t) in enumerate((("q", "bond price $q$"), ("d", "default intensity $d$"))):
            ax = axes[r, c]
            dd = np.abs(A[f] - Bs[f]) + 1e-12
            pm = ax.pcolormesh(BB, ZZ, dd, norm=LogNorm(vmin=1e-8, vmax=1e-1), cmap="Greys", shading="auto", rasterized=True)
            mf.draw_locus(ax, Bs["B"], Bs["z"], Bs["S"])
            ax.set_title(f"{t}, {lab}")
            if r == 1:
                ax.set_xlabel("debt $B$")
            if c == 0:
                ax.set_ylabel("log endowment $z$")
    fig.tight_layout(w_pad=1.0, h_pad=1.0, rect=(0, 0, 0.9, 1))
    cax = fig.add_axes([0.915, 0.15, 0.015, 0.7])
    cb = fig.colorbar(pm, cax=cax); cb.ax.tick_params(labelsize=7)
    cb.set_label("$|$fp32 $-$ fp64$|$", fontsize=8)
    mf.FIG = figdir
    mf.save(fig, "fp32_maps")


def paper_figs(figdir):
    mf.DATA, mf.FIG = os.path.join(ROOT, "ref", "data"), figdir
    import shutil
    for n in ("ou_slices", "ou_contours", "ou_irf", "ou_irf_states"):
        mf.ALL[n]()
        shutil.move(os.path.join(figdir, n + ".pdf"), os.path.join(figdir, n + "_paper.pdf"))


if __name__ == "__main__":
    d = sys.argv[1]
    os.makedirs(d, exist_ok=True)
    paper_figs(d)
    grid_slices(d)
    fp32_maps(d)
