"""MAT-01 check: basis truncation in the cleaner case (N_A = 1e18 cm^-3), where 90 subbands reach only 2.1-2.6 eV
above E_1. Single-particle 1->n oscillator strengths (m* = 0.19 m0 normalization) and their centroid over 0.1-2.5 eV
for 60, 90 and all subbands up to 3 eV (no response calculation, which would exceed the memory for 130 subbands)."""
from common import *

for ns in (3e13, 1e14):
    p = dict(BASE); p["NA"] = 1e18
    r0 = solve(ns, p, extend_to=0.0)
    z = r0["z"]; dz = z[1] - z[0]
    for nmax in (60, 90, 200):
        r = extend(r0, 3.0, nmax)
        E = r["E"] - r["E"][0]; psi = r["psi"]
        z1 = (psi[:, 0] * z) @ psi * dz
        f = 2 * 0.19 * 9.109e-31 * (E * 1.602e-19) * (z1 * 1e-10) ** 2 / 1.0546e-34 ** 2; f[0] = 0
        w = (E >= 0.1) & (E <= 2.5)
        print(f"ns {ns:.0e}: {len(E)} subbands up to {E[-1]:.2f} eV | f in window {f[w].sum():.4f} | f(>2 eV) {f[E > 2].sum():.4f} | "
              f"1->n centroid {np.sum(f[w]*E[w])/np.sum(f[w]):.4f} eV", flush=True)
