"""Write the solver inputs from the replication package: params.txt (the nine parameters x, the standard qf, its
band and lband, from data/ou_main.npz) and init.bin (the starting point of the paper's solve, data/calibration/start.npz:
int32 nB0, nz0; float64 B0, z0, V0, q0 with z fastest)."""
import os, sys
import numpy as np
ref = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/KM_partial_cuda/ref")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/KM_partial_cuda/inputs")
os.makedirs(out, exist_ok=True)
a = np.load(os.path.join(ref, "data", "ou_main.npz"))
with open(os.path.join(out, "params.txt"), "w") as f:
    f.write(" ".join(f"{v:.17g}" for v in a["x"]) + "\n")
    f.write(f"{float(a['qf']):.17g} {float(a['band']):.17g} {float(a['lband']):.17g}\n")
s = np.load(os.path.join(ref, "data", "calibration", "start.npz"))
with open(os.path.join(out, "init.bin"), "wb") as f:
    np.array([s["B"].size, s["z"].size], np.int32).tofile(f)
    for v in (s["B"], s["z"], s["V"], s["q"]):
        np.asarray(v, np.float64).tofile(f)
print("wrote", out)
