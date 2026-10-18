"""Convergence of the MAT-01 response with the number of subbands, the grid and the k mesh."""
from common import *
import time
HW = np.arange(0.15, 2.5, 0.0025)
def metrics(k):
    L = np.array([k.loss(h) for h in HW])
    i = int(np.argmax(L))
    cen = float(np.sum(HW * L) / np.sum(L))
    frac = float(np.sum(L[np.abs(HW - HW[i]) <= 0.05]) / np.sum(L))
    integ = float(np.sum(L / HW))          # ~ int Im p d omega (f-sum-like)
    return HW[i], cen, frac, integ
for NA in (5e19, 1e18):
    for ns in (2e13, 5e13):
        p = dict(BASE); p["NA"] = NA
        for nst, dzf, nk in ((10, 1, 300), (16, 1, 300), (24, 1, 300), (16, 0.5, 300), (16, 1, 600)):
            t = time.time()
            r = solve(ns, p, nstates=nst) if dzf == 1 else I.solve(ns, T=p["T"], NA=p["NA"], E_pin=p["E_pin"], well=materials(p)[0], cap=materials(p)[1], L=box_length(p), dz=(0.25 if box_length(p) <= 800 else 0.5), nstates=nst)
            out = []
            for kern in (False, True):
                k = response(r, p, kernel=kern, nk=nk)
                out.append(metrics(k))
            print(f"NA {NA:.0e} ns {ns:.0e} nst {nst} dz x{dzf} nk {nk}: E21 {r['E'][1]-r['E'][0]:.4f} | bare peak {out[0][0]:.3f} cen {out[0][1]:.3f} frac {out[0][2]:.2f} I {out[0][3]:.3e} | "
                  f"RPA peak {out[1][0]:.3f} cen {out[1][1]:.3f} frac {out[1][2]:.2f} I {out[1][3]:.3e} ({time.time()-t:.0f}s)", flush=True)
