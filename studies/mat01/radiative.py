"""MAT-01: radiative (superradiant) damping of the collective ISB modes of a single 2DEG sheet, omitted by the
quasi-static loss function. For a sheet in a medium of index n_r the radiative decay rate of a z-polarized mode with
2D spectral weight W (= n_s f / m0 summed over its constituent transitions, here from the k = 0 TDH eigenvectors of
meanmode.py) is Gamma = e^2 W sin^2(theta) / (2 eps0 c n_r cos(theta)) (theta: propagation angle inside the medium;
e.g. Vasanelli et al., PRL 125, 187401 (2020)). Reported: hbar*Gamma (eV, full width) of the strongest mode and the
value if all the weight sat in one mode (upper bound), at theta = 45 deg inside BSO (n_r = sqrt(eps_inf) = 2), and the
angle factor for other geometries, and the peak single-pass absorption of the strongest mode. Writes radiative.json."""
import json
from common import *

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; HBAR = 1.054571817e-34; C0 = 2.99792458e8; M0 = 9.1093837015e-31
out = []
nr = np.sqrt(BASE["eps_inf"])
ang = lambda th: np.sin(th) ** 2 / np.cos(th)
for NA in (5e19, 1e18):
    for ns in (5e12, 3e13, 1e14):
        p = dict(BASE); p["NA"] = NA
        r = solve(ns, p)
        z, psi, E = r["z"], r["psi"], r["E"]; dz = z[1] - z[0]
        nn = r["n_n"] * 1e4
        pairs = [(a, b) for a in range(len(E)) for b in range(a + 1, len(E)) if nn[a] - nn[b] > 1e-6 * nn.sum()]
        Ep = np.array([E[b] - E[a] for a, b in pairs]); Np = np.array([nn[a] - nn[b] for a, b in pairs])
        F = np.array([np.cumsum(psi[:, a] * psi[:, b]) * dz for a, b in pairs]).T
        K = (F.T @ F) * dz * 1e-10 * QE / (EPS0 * p["eps_inf"])
        Z = np.array([np.sum(psi[:, a] * z * psi[:, b]) * dz for a, b in pairs])
        s = np.sqrt(Ep * Np)
        w2, X = np.linalg.eigh(np.diag(Ep ** 2) + 2 * s[:, None] * K * s[None, :])
        Om = np.sqrt(np.clip(w2, 0, None))
        fc = (X.T @ (s * Z)) ** 2                                   # eV m^-2 A^2
        W = 2 * fc * QE * 1e-20 / HBAR ** 2                        # n_s f / m0 per mode (1/(kg m^2))
        g = HBAR * QE ** 2 * W / (2 * EPS0 * C0 * nr) / QE          # hbar Gamma (eV) per unit angle factor
        i = int(np.argmax(fc))
        o = dict(NA=NA, ns=ns, top_E=float(Om[i]), top_share=float(fc[i] / fc.sum()),
                 f_eff_total=float(W.sum() * M0 / (ns * 1e4)),       # total weight as n_s f / m0 -> f (relative to m0)
                 hG_top_45=float(g[i] * ang(np.pi / 4)), hG_all_45=float(g.sum() * ang(np.pi / 4)),
                 hG_top_per_angle=float(g[i]), angle_factor={"30": ang(np.pi / 6), "45": ang(np.pi / 4), "60": ang(np.pi / 3)})
        # peak single-pass absorption of the strongest mode at 45 deg for HWHM gamma = 10 and 30 meV (non-radiative full
        # width 2 gamma): A = 2 G_r G_nr / (G_r + G_nr)^2
        o["A_top_45"] = {f"{gm*1e3:.0f}": float(2 * o["hG_top_45"] * 2 * gm / (o["hG_top_45"] + 2 * gm) ** 2) for gm in (0.010, 0.030)}
        out.append(o)
        print(f"NA {NA:.0e} ns {ns:.1e}: strongest mode {o['top_E']:.3f} eV (share {o['top_share']:.2f}); total f (m0) {o['f_eff_total']:.2f}; "
              f"hbar Gamma_rad at 45 deg: strongest mode {o['hG_top_45']*1e3:.1f} meV, all weight in one mode {o['hG_all_45']*1e3:.1f} meV; "
              f"peak absorption of the strongest mode {o['A_top_45']}", flush=True)
json.dump(out, open(os.path.join(here, "radiative.json"), "w"), indent=1)
