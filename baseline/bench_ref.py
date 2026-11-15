"""Times the paper's own code (main calibration only) stage by stage and saves its outputs.
Stages: solve (implicit upwind time stepping from data/calibration/start.npz, strict tolerances),
ergodic density, moments (exact ergodic + 16 x 25,000-year simulation), impulse responses (irf.py).
Usage: python bench_ref.py OUTDIR [NB NZ]   (default: the paper grid, 201 x 75)
Dense-grid continuation (the rule of the report): KM_INIT=<ref_solution.npz of the next coarser grid> starts from that
solution, interpolated bilinearly; KM_DT_MAX and KM_TOLQ override the largest time step and the price tolerance."""
import os, sys, time, json
REF = os.path.expanduser("~/KM_partial_cuda/ref/code")
sys.path.insert(0, REF); os.chdir(REF)
import numpy as np
from cal_core import BMAX, NB, BAND, LBAND, RULE, START, on_grid, solve_level, SOLVER, STRICT
from calutil import unpack
from pdmodel import Model, ergodic_density, Result
from pdmoments import combined_moments, loss, simulate_paths, ergodic_moments, moments_paths
import pdtimestep
from generator import expected_paths, expected_paths_from_states

out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
a = dict(np.load(os.path.join(REF, "..", "data", "ou_main.npz")))
x, qf = a["x"].astype(float), float(a["qf"])
nz = int(sys.argv[3]) if len(sys.argv) > 3 else a["z"].size
NB = int(sys.argv[2]) if len(sys.argv) > 2 else NB
s = dict(np.load(os.environ.get("KM_INIT", START)))
par, zp = unpack(x, qf, nz, BAND, LBAND)
m = Model(par, zp, 0.0, BMAX, NB)
if s["V"].shape[0] == m.nB:
    V0, q0 = on_grid(m, (s["V"], s["q"], s["z"]))
else:   # another debt grid: the start interpolated bilinearly (in z, then in B), as in bin/km
    Bs = s["B"] if "B" in s else np.linspace(0.0, BMAX, s["V"].shape[0])
    bil = lambda F: np.stack([np.interp(m.B, Bs, col) for col in np.stack([np.interp(m.z, s["z"], F[i]) for i in range(Bs.size)]).T], 1)
    V0, q0 = bil(s["V"]), np.clip(bil(s["q"]), 1e-6, 1.0)
T = {}
t0 = time.perf_counter()
SOLVER, STRICT = dict(SOLVER), dict(STRICT)
if "KM_DT_MAX" in os.environ: SOLVER["dt_max"] = float(os.environ["KM_DT_MAX"])
if "KM_TOLQ" in os.environ: STRICT["tolq"] = float(os.environ["KM_TOLQ"])
res = pdtimestep.solve_timestep(m, V0, q0, print_every=10**9, verbose=False, ergodic=False, **SOLVER, **STRICT)
T["solve"] = time.perf_counter() - t0
t0 = time.perf_counter(); g = ergodic_density(m, res.A); T["ergodic"] = time.perf_counter() - t0
res.g = g
t0 = time.perf_counter(); E = ergodic_moments(res); T["ergodic_moments"] = time.perf_counter() - t0
t0 = time.perf_counter(); sims = simulate_paths(res, n_paths=16, T_years=25000, steps_per_year=100, seed=1); T["simulate"] = time.perf_counter() - t0
t0 = time.perf_counter(); Sm, Ss = moments_paths(sims); T["path_moments"] = time.perf_counter() - t0
sol = dict(V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=g, B=m.B, z=m.z, x=x, qf=qf, band=BAND, lband=LBAND, resV=res.resV, resq=res.resq)
t0 = time.perf_counter()
tt, irf = expected_paths(sol, T=21.0)
EB = float((g.sum(1) / g.sum() * m.B).sum())
tt2, irs, ybar = expected_paths_from_states(sol, [(mm * EB, 0.0) for mm in [1.0, 0.75, 1.25]], T=21.0)
T["irf"] = time.perf_counter() - t0
np.savez(os.path.join(out, "ref_solution.npz"), **sol, frozen=res.frozen, which=res.which, iters=res.iters)
np.savez(os.path.join(out, "ref_irf.npz"), t=tt, **{f"{k}_{f}": v for k in irf for f, v in irf[k].items()},
         **{f"{i}_{k}_{f}": v for i, r in enumerate(irs) for k in r for f, v in r[k].items()}, EB=EB, ybar=ybar)
M = {k: E[k] for k in ["debt_y_mean","debt_y_sd","ds_y_mean","ds_y_sd","dd_y_mean","spread_sd","spread_mean","corr_spread_y","corr_spread_debt"]}
M.update({k: Sm[k] for k in ["pd_freq","pd_mean_cond","pd_sd_cond","y_persist","y_sd","dc_y_mean_cond","dc_y_sd_cond","c_sd_rel"]})
info = dict(times=T, iters=int(res.iters), resV=float(res.resV), resq=float(res.resq), converged=bool(res.converged),
            n_frozen=int(res.n_frozen), passes=int(res.passes), direct=int(res.direct_solves),
            moments=M, loss=float(loss(M)), ergodic=E, sim_mean=Sm, sim_sd=Ss)
json.dump(info, open(os.path.join(out, "ref_info.json"), "w"), indent=1, default=float)
print(json.dumps(dict(times=T, iters=info["iters"], resV=info["resV"], resq=info["resq"], loss=info["loss"], frozen=info["n_frozen"], passes=info["passes"], direct=info["direct"]), default=float))
