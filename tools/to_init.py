"""Write a solver starting point (init.bin: int32 nB0, nz0; float64 B0, z0, V0, q0 with z fastest) from a result
folder of bin/km, for continuation from a coarser grid.  Usage: python tools/to_init.py <result_dir> <init.bin>"""
import os, sys
import numpy as np
r, out = sys.argv[1], sys.argv[2]
B, z, V, q = (np.load(os.path.join(r, f + ".npy")) for f in ("B", "z", "V", "q"))
with open(out, "wb") as f:
    np.array([B.size, z.size], np.int32).tofile(f)
    for v in (B, z, V, q):
        np.asarray(v, np.float64).tofile(f)
