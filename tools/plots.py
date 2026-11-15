"""plots.py: draw the paper's exhibits from the results written by bin/km.

The figures are drawn by the replication package's own plotting functions (ref/code/make_figures.py:
fig_ou_slices, fig_ou_contours, fig_irf, fig_irf_states), with the same matplotlib settings, fed with the
arrays that bin/km wrote to disk (.npy files and meta.json). For each result folder a data folder in the
package's format (ou_main.npz, irf.npz, irf_states.npz) is assembled and the package functions are pointed at it.
On denser grids the display smoothing of the package, set in grid steps, is rescaled to the same width in (B, z).

Usage: python tools/plots.py <result_dir> <figure_dir> [suffix]
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = os.path.join(ROOT, "ref")
sys.path.insert(0, os.path.join(REF, "code"))
import make_figures as mf  # noqa: E402  (sets the package's rcParams, pgf backend)
import scipy.ndimage as ndi  # noqa: E402

KEYS = ["y", "yn", "c", "d", "B", "spread", "issue", "defc"]
NB0, NZ0 = 201, 75   # the paper's grid


def to_package(res, data):
    """Write ou_main.npz, irf.npz and irf_states.npz in the replication package's layout."""
    ref = np.load(os.path.join(REF, "data", "ou_main.npz"))
    meta = json.load(open(os.path.join(res, "meta.json")))
    arr = {k: np.load(os.path.join(res, k + ".npy")) for k in ("B", "z", "V", "q", "S", "c", "d", "g")}
    np.savez(os.path.join(data, "ou_main.npz"), **arr, x=ref["x"], names=ref["names"], qf=ref["qf"], band=ref["band"],
             lband=ref["lband"], resV=meta["resV"], resq=meta["resq"], loss=meta.get("loss", np.nan))
    if not os.path.exists(os.path.join(res, "irf.npy")):
        return arr
    irf = np.load(os.path.join(res, "irf.npy"))     # [run, field, t]: 0 ergodic shock, 1+2i / 2+2i state i shock /
    t = np.load(os.path.join(res, "irf_t.npy"))     # control, 7 ergodic control
    np.savez(os.path.join(data, "irf.npz"), t=t, **{f"shock_{k}": irf[0, f] for f, k in enumerate(KEYS)},
             **{f"control_{k}": irf[7, f] for f, k in enumerate(KEYS)})
    g = arr["g"] / arr["g"].sum()
    EB = float((g.sum(1) * arr["B"]).sum())
    ybar = float(irf[7, 1, 0])
    st = {}
    for i in range(3):
        for f, k in enumerate(KEYS):
            st[f"{i}_shock_{k}"] = irf[1 + 2 * i, f]
            st[f"{i}_control_{k}"] = irf[2 + 2 * i, f]
    np.savez(os.path.join(data, "irf_states.npz"), t=t, EB=EB, ybar=ybar, **st)
    return arr


class grid_smoothing:
    """Rescale the package's display smoothing (in grid steps) to the same width in (B, z) on another grid."""
    def __init__(self, nB, nz):
        self.fB = (nB - 1) / (NB0 - 1); self.fz = (nz - 1) / (NZ0 - 1)

    def __enter__(self):
        self.g, self.g1 = ndi.gaussian_filter, ndi.gaussian_filter1d
        fB, fz, g, g1 = self.fB, self.fz, self.g, self.g1

        def gf(x, sigma, **kw):
            if np.ndim(x) == 2 and np.ndim(sigma) == 1:
                sigma = (sigma[0] * fB, sigma[1] * fz)
            return g(x, sigma, **kw)

        def gf1(x, sigma, **kw):   # the zero-drift locus is smoothed with sigma 1 across endowment rows
            if sigma == 1.0:
                sigma = sigma * fz
            return g1(x, sigma, **kw)
        ndi.gaussian_filter, ndi.gaussian_filter1d = gf, gf1
        return self

    def __exit__(self, *a):
        ndi.gaussian_filter, ndi.gaussian_filter1d = self.g, self.g1


def draw(res, figdir, suffix="", names=("ou_slices", "ou_contours", "ou_irf", "ou_irf_states")):
    os.makedirs(figdir, exist_ok=True)
    data = tempfile.mkdtemp()
    arr = to_package(res, data)
    mf.DATA, mf.FIG = data, figdir
    with grid_smoothing(arr["B"].size, arr["z"].size):
        for n in names:
            if n.startswith("ou_irf") and not os.path.exists(os.path.join(data, "irf.npz")):
                continue
            mf.ALL[n]()
            if suffix:
                shutil.move(os.path.join(figdir, n + ".pdf"), os.path.join(figdir, f"{n}{suffix}.pdf"))
    shutil.rmtree(data)


if __name__ == "__main__":
    draw(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
