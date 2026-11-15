"""tables.py: the LaTeX tables of the report, from the meta.json and .npy files written by bin/km and from the
paper-code baseline (baseline/run1: paper grid).  Usage: python tools/tables.py <outdir>"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "report", "tables")
os.makedirs(OUT, exist_ok=True)
STAGES = ["solve", "ergodic", "moments", "irf"]


def meta(name):
    return json.load(open(os.path.join(RES, name, "meta.json")))


def times(name):
    """Median over the repetitions of each stage time (moments = ergodic moments + simulation + path moments)."""
    ms = [meta(name)] + [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, name, "reps", "meta_r*.json")))]
    rows = []
    for m in ms:
        t = m["times"]
        rows.append([t["solve"], t["ergodic"], t["ergodic_moments"] + t.get("simulate", 0) + t.get("path_moments", 0), t.get("irf", 0)])
    return np.median(np.array(rows), 0)


def ref_times(d):
    t = json.load(open(os.path.join(ROOT, "baseline", d, "ref_info.json")))["times"]
    return np.array([t["solve"], t["ergodic"], t["ergodic_moments"] + t["simulate"] + t["path_moments"], t["irf"]])


def fmt_t(x):
    return f"{x:.3f}" if x < 1 else (f"{x:.2f}" if x < 10 else f"{x:.1f}")


def sci(x, nd=1):
    if x == 0:
        return "0"
    m, e = f"{x:.{nd}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


# ------------------------------------------------------------------ run times
def runtime_table():
    lines = []
    blocks = [
        ("Paper grid, $201\\times75$", "run1", [
            ("p_cpu64_exact", "CPU, fp64, exact line sweeps"), ("p_cpu64", "CPU, fp64"), ("p_gpu64", "GPU, fp64"),
            ("p_cpu32", "CPU, fp32"), ("p_gpu32", "GPU, fp32")]),
        ("Dense grid, $500\\times121$", None, [
            ("d500_cpu64_exact", "CPU, fp64, exact line sweeps"), ("d500_cpu64", "CPU, fp64"), ("d500_gpu64", "GPU, fp64"),
            ("d500_cpu32", "CPU, fp32"), ("d500_gpu32", "GPU, fp32")]),
        ("Dense grid, $1000\\times121$ (continuation from $500\\times121$)", None, [
            ("d1000_cpu64", "CPU, fp64"), ("d1000_gpu64", "GPU, fp64"), ("d1000_cpu32", "CPU, fp32"), ("d1000_gpu32", "GPU, fp32")]),
        ("Dense grid, $2000\\times121$ (continuation from $1000\\times121$)", None, [
            ("d2000_cpu64", "CPU, fp64"), ("d2000_gpu64", "GPU, fp64"), ("d2000_cpu32", "CPU, fp32"), ("d2000_gpu32", "GPU, fp32")]),
    ]
    for title, refd, rows in blocks:
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{title}}}}}\\\\")
        tref = None
        if refd and os.path.exists(os.path.join(ROOT, "baseline", refd, "ref_info.json")):
            tref = ref_times(refd); ri = json.load(open(os.path.join(ROOT, "baseline", refd, "ref_info.json")))
            lines.append("\\quad Paper code (Python, Numba, SciPy) & " + f"{ri['iters']} & " +
                         " & ".join(fmt_t(x) for x in tref) + f" & {fmt_t(tref.sum())} & 1 & 1\\\\")
        for name, lab in rows:
            if not os.path.exists(os.path.join(RES, name, "meta.json")):
                continue
            m = meta(name); t = times(name)
            sp = (f"{tref[0] / t[0]:.0f}" if tref is not None else "--")
            spt = (f"{tref.sum() / t.sum():.0f}" if tref is not None else "--")
            lines.append(f"\\quad {lab} & {m['iters']} & " + " & ".join(fmt_t(x) for x in t) + f" & {fmt_t(t.sum())} & {sp} & {spt}\\\\")
        lines.append("\\addlinespace")
    open(os.path.join(OUT, "runtimes.tex"), "w").write("\n".join(lines[:-1]) + "\n")


# ------------------------------------------------------------------ verification against the paper code
FIELDS = ["V", "q", "S", "c", "d"]


def diffs(name, refdir):
    r = np.load(os.path.join(ROOT, "baseline", refdir, "ref_solution.npz"))
    L = lambda f: np.load(os.path.join(RES, name, f + ".npy"))
    g = r["g"] / r["g"].sum()
    out = {}
    for f in FIELDS:
        dd = np.abs(L(f) - r[f]); out[f] = (dd.max(), (dd * g).sum())
    out["g"] = np.abs(L("g") / L("g").sum() - g).sum()
    out["frozen"] = int((L("frozen") != r["frozen"]).sum())
    ri = np.load(os.path.join(ROOT, "baseline", refdir, "ref_irf.npz"))
    irf = L("irf"); keys = ["y", "yn", "c", "d", "B", "spread", "issue", "defc"]
    ref = np.stack([np.stack([ri[f"shock_{k}"] for k in keys])] + [np.stack([ri[f"{i}_{s}_{k}" ] for k in keys]) for i in range(3) for s in ("shock", "control")])
    n = min(ref.shape[-1], irf.shape[-1])
    out["irf"] = np.abs(irf[:7, :, :n] - ref[:, :, :n]).max()
    return out


def verification_table():
    lines = []
    for title, refd, rows in [("Paper grid, $201\\times75$", "run1", [("p_cpu64_exact", "CPU, fp64, exact sweeps"), ("p_cpu64", "CPU, fp64"), ("p_gpu64", "GPU, fp64"), ("p_cpu32", "CPU, fp32"), ("p_gpu32", "GPU, fp32")])]:
        if not os.path.exists(os.path.join(ROOT, "baseline", refd, "ref_irf.npz")):
            continue
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for name, lab in rows:
            if not os.path.exists(os.path.join(RES, name, "meta.json")):
                continue
            d = diffs(name, refd); m = meta(name)
            lines.append(f"\\quad {lab} & {m['n_frozen']} ({d['frozen']}) & " + " & ".join(sci(d[f][0]) for f in FIELDS) +
                         f" & {sci(d['q'][1])} & {sci(d['g'])} & {sci(d['irf'])}\\\\")
        lines.append("\\addlinespace")
    open(os.path.join(OUT, "verification.tex"), "w").write("\n".join(lines[:-1]) + "\n")


# ------------------------------------------------------------------ Table 2
ROWS = [("Partial default: frequency", "pd_freq", 1, 0.36, 0.37, 0.39, 2), ("Partial default: mean $\\mid d>0$", "pd_mean_cond", 1, 0.38, 0.39, 0.48, 2),
        ("Partial default: s.d. $\\mid d>0$", "pd_sd_cond", 1, 0.22, 0.19, 0.26, 2), ("Debt to output: mean", "debt_y_mean", 1, 0.32, 0.32, 0.30, 2),
        ("Debt to output: s.d.", "debt_y_sd", 1, 0.18, 0.25, 0.20, 2), ("Debt service to output: mean", "ds_y_mean", 1, 0.036, 0.035, 0.033, 3),
        ("Debt service to output: s.d.", "ds_y_sd", 1, 0.021, 0.022, 0.018, 3), ("Debt due to output: mean", "dd_y_mean", 3, 0.049, 0.050, 0.050, 3),
        ("Spread: s.d.", "spread_sd", 1.25, 0.041, 0.037, 0.029, 3), ("Output: persistence", "y_persist", 1, 0.89, 0.88, 0.93, 2),
        ("Output: s.d.", "y_sd", 1, 0.10, 0.12, 0.105, 3)]
UNT = [("Defaulted coupons to output: mean $\\mid d>0$", "dc_y_mean_cond", 0.052, 0.040, 0.045, 3), ("Defaulted coupons to output: s.d. $\\mid d>0$", "dc_y_sd_cond", 0.064, 0.036, 0.034, 3),
       ("Spread: mean", "spread_mean", 0.053, 0.016, 0.020, 3), ("Spread: correlation with output", "corr_spread_y", -0.17, -0.38, -0.76, 2),
       ("Spread: correlation with debt", "corr_spread_debt", 0.24, 0.56, 0.84, 2), ("Consumption s.d. relative to output s.d.", "c_sd_rel", 1.00, 0.91, 0.93, 2)]


def moments_table(cols, fname):
    """cols: list of (header, moments dict, loss)."""
    def f(v, nd):
        s = f"{v:.{nd}f}"
        return s.replace("-", "$-$")
    lines = []
    lines.append("\\multicolumn{%d}{l}{\\textit{Targeted}}\\\\" % (5 + len(cols)))
    for lab, k, w, dat, amr, pap, nd in ROWS:
        w_ = f"{w:g}"
        lines.append(f"\\quad {lab} & {w_} & {f(dat, nd)} & {f(amr, nd)} & {f(pap, nd)} & " + " & ".join(f(c[1][k], nd) for c in cols) + "\\\\")
    lines.append("\\multicolumn{%d}{l}{\\textit{Untargeted}}\\\\" % (5 + len(cols)))
    for lab, k, dat, amr, pap, nd in UNT:
        lines.append(f"\\quad {lab} & & {f(dat, nd)} & {f(amr, nd)} & {f(pap, nd)} & " + " & ".join(f(c[1][k], nd) for c in cols) + "\\\\")
    lines.append("\\addlinespace")
    lines.append("\\quad Loss & & & 0.228 & 0.259 & " + " & ".join(f"{c[2]:.3f}" for c in cols) + "\\\\")
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")


def main():
    runtime_table()
    verification_table()
    ref = json.load(open(os.path.join(ROOT, "baseline", "run1", "ref_info.json")))
    cols = [("paper code", ref["moments"], ref["loss"])]
    for n in ["p_cpu64", "p_gpu32"]:
        m = meta(n); cols.append((n, m["moments"], m["loss"]))
    moments_table(cols, "table2.tex")
    cols = [("p", meta("p_cpu64")["moments"], meta("p_cpu64")["loss"])]
    for n in ["d500_cpu64", "d1000_cpu64", "d2000_cpu64"]:
        if os.path.exists(os.path.join(RES, n, "meta.json")):
            m = meta(n); cols.append((n, m["moments"], m["loss"]))
    moments_table(cols, "table2_dense.tex")
    # fp32 against fp64 of the same back end and grid
    lines = []
    for title, pairs in [("$201\\times75$", [("p_cpu32", "p_cpu64", "CPU"), ("p_gpu32", "p_gpu64", "GPU")]),
                         ("$500\\times121$", [("d500_cpu32", "d500_cpu64", "CPU"), ("d500_gpu32", "d500_gpu64", "GPU")]),
                         ("$1000\\times121$", [("d1000_cpu32", "d1000_cpu64", "CPU"), ("d1000_gpu32", "d1000_gpu64", "GPU")]),
                         ("$2000\\times121$", [("d2000_cpu32", "d2000_cpu64", "CPU"), ("d2000_gpu32", "d2000_gpu64", "GPU")])]:
        for a, b, lab in pairs:
            if not (os.path.exists(os.path.join(RES, a, "meta.json")) and os.path.exists(os.path.join(RES, b, "meta.json"))):
                continue
            La = lambda f: np.load(os.path.join(RES, a, f + ".npy")); Lb = lambda f: np.load(os.path.join(RES, b, f + ".npy"))
            g = Lb("g") / Lb("g").sum()
            ma, mb = meta(a), meta(b)
            cells = []
            for f in FIELDS:
                dd = np.abs(La(f) - Lb(f)); cells += [sci(dd.max()), sci((dd * g).sum())]
            gl1 = np.abs(La("g") / La("g").sum() - g).sum()
            lines.append(f"{title} & {lab} & {ma['iters']} & {sci(ma['resV'])} & {sci(ma['resq'])} & {cells[0]} & {cells[2]} & {cells[3]}"
                         f" & {cells[6]} & {cells[8]} & {cells[9]} & {sci(gl1)} & " + f"{(ma['loss'] - mb['loss']) * 1e4:+.0f}".replace("-", "$-$") + "\\\\")
    open(os.path.join(OUT, "fp32.tex"), "w").write("\n".join(lines) + "\n")
    # convergence summary of every run, in the order of the run-time table
    lines = []
    order = [("p", "$201\\times75$"), ("d500", "$500\\times121$"), ("d1000", "$1000\\times121$"), ("d2000", "$2000\\times121$")]
    labs = [("cpu64_exact", "CPU, fp64, exact sweeps"), ("cpu64", "CPU, fp64"), ("gpu64", "GPU, fp64"), ("cpu32", "CPU, fp32"), ("gpu32", "GPU, fp32")]
    DTMAX = {"p": 0.5, "d500": 0.5, "d1000": 0.25, "d2000": 0.125}
    for pre, title in order:
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for suf, lab in labs:
            pth = os.path.join(RES, f"{pre}_{suf}", "meta.json")
            if not os.path.exists(pth):
                continue
            m = json.load(open(pth))
            crit = "$64\\varepsilon_{32}$" if m["fp"] == 32 else f"{sci(m['tolV'])}, {sci(m['tolq'])}"
            stop = "tolerance" if m["converged"] else ("stagnation" if m.get("stalled") else "step cap")
            lines.append(f"\\quad {lab} & {m['iters']} & {DTMAX[pre]} & {crit} & {sci(m['resV'], 2)} & {sci(m['resq'], 2)} & {m['n_frozen']} & {stop}\\\\")
        lines.append("\\addlinespace")
    open(os.path.join(OUT, "convergence.tex"), "w").write("\n".join(lines[:-1]) + "\n")
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
