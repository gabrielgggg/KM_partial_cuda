"""Compare a solver output folder with the paper code's reference solution (baseline/run1/ref_solution.npz)."""
import json, os, sys
import numpy as np
d = sys.argv[1]
ref = np.load(sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/KM_partial_cuda/baseline/run1/ref_solution.npz"))
L = lambda k: np.load(os.path.join(d, k + ".npy"))
meta = json.load(open(os.path.join(d, "meta.json")))
g = L("g"); g = g / g.sum(); gr = ref["g"] / ref["g"].sum()
out = {}
for k in ["V", "q", "S", "c", "d"]:
    a, b = L(k), ref[k]
    out[k] = (float(np.abs(a - b).max()), float((gr * np.abs(a - b)).sum()))
out["g_L1"] = float(np.abs(g - gr).sum())
out["frozen_diff"] = int((L("frozen").astype(bool) != ref["frozen"]).sum())
print(json.dumps(dict(iters=meta["iters"], resV=meta["resV"], resq=meta["resq"], frozen=meta["n_frozen"], times=meta["times"], loss=meta["loss"]), indent=None))
for k, v in out.items(): print(k, v)
irf_path = os.path.join(d, "irf.npy")
if os.path.exists(irf_path):
    I = np.load(irf_path)
    R = np.load(os.path.join(os.path.dirname(sys.argv[2]) if len(sys.argv) > 2 else os.path.expanduser("~/KM_partial_cuda/baseline/run1"), "ref_irf.npz"))
    F = ["y", "yn", "c", "d", "B", "spread", "issue", "defc"]
    keys = ["shock", "0_shock", "0_control", "1_shock", "1_control", "2_shock", "2_control", "control"]
    md = 0.0
    for r, kk in enumerate(keys):
        for f, nm in enumerate(F):
            md = max(md, float(np.abs(I[r, f] - R[f"{kk}_{nm}"]).max()))
    print("irf max abs diff", md)
