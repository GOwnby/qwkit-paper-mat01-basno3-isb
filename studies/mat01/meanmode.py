"""MAT-01: linewidth-independent location of the absorption. The k = 0 time-dependent Hartree problem with the
background eps_inf (the phonon correction to the depolarization is below 1%, Appendix A) is solved as an eigenvalue
problem, Omega^2 = E^2 + 2 sqrt(E N) K sqrt(E N), over all pairs of subbands with an occupation difference (E: pair
energies, N: sheet-density differences, K = e S / (eps0 eps_inf)); mode oscillator strengths follow from the
eigenvectors. Reported: the oscillator-strength-weighted mean energy of the collective modes and of the bare
transitions (the first moment of the absorbed power, free of any linewidth), their ratio, the share of the strongest
mode, the 1->2 energy dressed alone (E21 sqrt(1 + alpha)), and the f-sum check. Writes meanmode.json."""
import json
from common import *

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12
NSL = [5e12, 1e13, 1.5e13, 2e13, 3e13, 4e13, 5e13, 6e13, 7e13, 8e13, 1e14, 1.2e14]
out = []
for NA in (5e19, 1e18):
    for ns in NSL:
        p = dict(BASE); p["NA"] = NA
        r = solve(ns, p)
        z, psi, E = r["z"], r["psi"], r["E"]; dz = z[1] - z[0]
        nn = r["n_n"] * 1e4                                   # m^-2
        pairs = [(a, b) for a in range(len(E)) for b in range(a + 1, len(E)) if nn[a] - nn[b] > 1e-6 * nn.sum()]
        Ep = np.array([E[b] - E[a] for a, b in pairs]); Np = np.array([nn[a] - nn[b] for a, b in pairs])
        F = np.array([np.cumsum(psi[:, a] * psi[:, b]) * dz for a, b in pairs]).T           # (Nz, P)
        K = (F.T @ F) * dz * 1e-10 * QE / (EPS0 * p["eps_inf"])                             # eV m^2
        Z = np.array([np.sum(psi[:, a] * z * psi[:, b]) * dz for a, b in pairs])            # A
        s = np.sqrt(Ep * Np)
        M = np.diag(Ep ** 2) + 2 * s[:, None] * K * s[None, :]
        w2, X = np.linalg.eigh(M)
        Om = np.sqrt(np.clip(w2, 0, None))
        fb = Ep * Np * Z ** 2                                  # bare oscillator strengths (common factor dropped)
        fc = (X.T @ (s * Z)) ** 2                              # collective
        eb = float(np.sum(fb * Ep) / fb.sum()); ec = float(np.sum(fc * Om) / fc.sum())
        i12 = pairs.index((0, 1))
        E12d = float(np.sqrt(Ep[i12] ** 2 + 2 * s[i12] ** 2 * K[i12, i12]))
        order = np.argsort(-fc)[:8]
        o = dict(NA=NA, ns=ns, E21=float(E[1] - E[0]), mean_bare=eb, mean_coll=ec, ratio=ec / eb,
                 modes_E=[float(Om[i]) for i in order], modes_w=[float(fc[i] / fc.sum()) for i in order],
                 top_share=float(fc.max() / fc.sum()), top_E=float(Om[int(np.argmax(fc))]), E12_dressed=E12d,
                 fsum=float(fc.sum() / fb.sum()), n_pairs=len(pairs))
        out.append(o)
        print(f"NA {NA:.0e} ns {ns:.1e}: E21 {o['E21']:.3f} | mean energy bare {eb:.3f} collective {ec:.3f} ratio {o['ratio']:.2f} | "
              f"strongest mode {o['top_E']:.3f} eV share {o['top_share']:.2f} | 1->2 alone dressed {E12d:.3f} | f-sum {o['fsum']:.4f} "
              f"| pairs {len(pairs)}", flush=True)
json.dump(out, open(os.path.join(here, "meanmode.json"), "w"), indent=1)
