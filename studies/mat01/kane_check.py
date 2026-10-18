"""MAT-01 check: dipoles of the two-component Kane states (core.kane_dipoles, exact orthogonality, no Loewdin step)
against the Loewdin-orthonormalized one-band states used in the studies. n_s = 1e13, 3e13, 1e14 cm^-2, both
acceptor densities, 300 K. Prints z12, z13, the 1->n oscillator-strength sum and single-particle 1->n centroid."""
from common import *
from qwkit import core as q

for NA in (5e19, 1e18):
    for ns in (1e13, 3e13, 1e14):
        p = dict(BASE); p["NA"] = NA
        r = solve(ns, p)
        z, psi, E, V = r["z"], r["psi"], r["E"], r["V"]; dz = z[1] - z[0]
        Er, pr = q.solve_states(z, V, r["m_edge"], r["Eg"], nstates=len(E), tol=1e-8)
        pr = pr / np.sqrt(np.sum(pr ** 2, axis=0) * dz)
        Z2, ov = q.kane_dipoles(z, pr, Er, V, r["m_edge"], r["Eg"])
        ZL = q.dipoles(z, psi)
        zh, phi = q.kane_valence(z, pr, Er, V, r["m_edge"], r["Eg"])
        vw = np.sum(phi[:, :2] ** 2, axis=0) * dz
        fL = np.array([(E[j] - E[0]) * ZL[0, j] ** 2 for j in range(1, len(E))])
        fK = np.array([(E[j] - E[0]) * Z2[0, j] ** 2 for j in range(1, len(E))])
        print(f"NA {NA:.0e} ns {ns:.0e}: n {len(E)} two-band overlap {ov:.1e} valence weight {vw.round(3)} | z12 Loewdin "
              f"{ZL[0,1]:.3f} Kane {Z2[0,1]:.3f} | z13 {ZL[0,2]:.3f} {Z2[0,2]:.3f} | f-sum ratio {fK.sum()/fL.sum():.4f} | "
              f"1->n centroid {np.sum(fL*(E[1:]-E[0]))/fL.sum():.4f} {np.sum(fK*(E[1:]-E[0]))/fK.sum():.4f}", flush=True)
