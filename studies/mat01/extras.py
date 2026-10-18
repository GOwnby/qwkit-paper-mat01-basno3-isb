"""MAT-01 supporting numbers quoted in the text, logged here:
 (1) accuracy of the fitted in-plane nonparabolicity G_n against the exact subband dispersion (subbands 1-2, k up to
     the Fermi wavevector + 0.3 eV);
 (2) share of the three polar modes in the edge 2 -> 1 LO rate;
 (3) hbar Q v_F / hbar w_LO for the emission wavevector (dynamic vs static screening);
 (4) energy reached by the 90-subband basis;
 (5) sensitivity of the self-consistent solution to the mixing at 5e12 cm^-2 (compensated);
 (6) depolarization coupling among the 1 -> n transitions: the Casida-type matrix Omega^2 = E^2 + 2 sqrt(E dn) K
     sqrt(E dn) at k = 0, its largest-eigenvalue share of the coupling and the normalized overlaps of the transition
     densities (why the strength does not condense into one multisubband plasmon)."""
from common import *

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; HBAR = 1.054571817e-34; M0 = 9.1093837015e-31
C = krpa.froehlich_weights(BASE["eps_inf"], MODES)
for NA in (5e19, 1e18):
    for ns in (5e12, 1e13, 3e13, 1e14):
        p = dict(BASE); p["NA"] = NA
        r = solve(ns, p)
        z, psi, E, V = r["z"], r["psi"], r["E"], r["V"]; dz = z[1] - z[0]
        # (1) dispersion fit
        errs = []
        for n in (0, 1):
            emax = max(r["mu"] - E[n], 0.0) + 0.3
            x0 = np.linspace(0.02, r["m_n"][n] * emax * (1 + emax / r["G_n"][n]), 6)
            ex = I.subband_dispersion(z, V, r["m_edge"], r["Eg"], n, E[n], x0)
            fit = I.dispersion(x0 / r["m_n"][n], r["m_n"][n], r["G_n"][n])
            errs.append(float(np.max(np.abs(fit / ex - 1))))
        # (2) mode shares, (3) dynamic-screening ratio
        rates = [sc.lo_rate_modes(z, psi[:, 1], psi[:, 0], E[1] - E[0], [(l, c)], T=r["T"], m_eff=r["m_n"][0]) for (t, l), c in zip(MODES, C)]
        sh = np.array(rates) / np.sum(rates)
        m1 = r["m_n"][0] * M0; ef = max(r["mu"] - E[0], 1e-4)
        vF = np.sqrt(2 * ef * QE / m1)
        rat = [np.sqrt(2 * m1 * (E[1] - E[0] - l) * QE) * vF / (l * QE) for (t, l) in MODES[1:] if E[1] - E[0] > l]
        # (6) coupling of 1 -> n transitions at k = 0
        nn = min(12, len(E))
        dn = (r["n_n"][0] - r["n_n"][1:nn]) * 1e4                     # m^-2
        Et = E[1:nn] - E[0]
        F = np.cumsum(psi[:, :1] * psi[:, 1:nn], axis=0) * dz          # F_{1n}(z)
        K = (F.T @ F) * dz * 1e-10 * QE / (EPS0 * BASE["eps_inf"])     # eV m^2 (times dn gives eV)
        A = np.sqrt(np.outer(Et * dn, Et * dn)) * K * 2               # eV^2
        w, U = np.linalg.eigh(A)
        Cn = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
        Om = np.sqrt(np.clip(np.linalg.eigvalsh(np.diag(Et ** 2) + A), 0, None))
        Z1 = (psi[:, 0] * z) @ psi[:, 1:] * dz
        f1n = Et * Z1[:nn - 1] ** 2 * (r["n_n"][0] - r["n_n"][1:nn])
        share12 = float(f1n[0] / np.sum(f1n))
        print(f"NA {NA:.0e} ns {ns:.0e}: G-fit max error {max(errs):.2%} | basis {len(E)} subbands to {E[-1]-E[0]:.2f} eV | "
              f"LO shares {np.round(sh, 3)} | hbar Q vF / hbar wLO {np.round(rat, 1)} | coupling: top eigenvalue share "
              f"{w[-1]/w.sum():.2f}, overlaps C12-13 {Cn[0,1]:+.2f} C12-14 {Cn[0,2]:+.2f} C13-14 {Cn[1,2]:+.2f} | "
              f"E_n1 {np.round(Et[:4], 3)} -> Omega {np.round(np.sort(Om)[:4], 3)} | 1->2 share of 1->n strength {share12:.2f}", flush=True)
# (5) mixing sensitivity
p = dict(BASE); p["NA"] = 5e19
ra = solve(5e12, p, extend_to=0.0); rb = solve(5e12, p, extend_to=0.0, mix=0.1, maxit=600)
print(f"mixing test 5e12 (compensated): residual {ra['err']:.1e} / {rb['err']:.1e} eV; E21 {ra['E'][1]-ra['E'][0]:.5f} / "
      f"{rb['E'][1]-rb['E'][0]:.5f}; E_F - E1 {ra['mu']-ra['E'][0]:.5f} / {rb['mu']-rb['E'][0]:.5f}", flush=True)
