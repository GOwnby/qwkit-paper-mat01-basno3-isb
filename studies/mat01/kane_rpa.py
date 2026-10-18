"""MAT-01 check: collective response with the two-band Kane metric carried through. The transition densities that
enter the Hartree kernel and the dipoles are those of the two-component states, rho_ab = (psi_a psi_b + phi_a phi_b)
/ (N_a N_b) with phi the valence components (core.kane_valence, interpolated to the grid) and N_a^2 the two-band norm;
no Loewdin step. Compared with the baseline (Loewdin-orthonormalized one-band envelopes). Medians of the single-
particle and collective absorption over 0.10-2.5 eV, gamma = 10 meV, 300 K."""
from common import *
from qwkit import core as q

HW = np.arange(0.10, 2.5001, 0.005)


def kane_ops(r):
    z, V = r["z"], r["V"]; dz = z[1] - z[0]
    E, P = q.solve_states(z, V, r["m_edge"], r["Eg"], nstates=len(r["E"]), tol=1e-8)
    P = P / np.sqrt(np.sum(P ** 2, axis=0) * dz)
    zh, phi = q.kane_valence(z, P, E, V, r["m_edge"], r["Eg"])
    ph = np.zeros_like(P); ph[1:-1] = 0.5 * (phi[1:] + phi[:-1])          # valence components on the cell centres
    N = np.sqrt(np.sum(P ** 2, axis=0) * dz + np.sum(phi ** 2, axis=0) * dz)
    rho = (P[:, :, None] * P[:, None, :] + ph[:, :, None] * ph[:, None, :]) / (N[:, None] * N[None, :])[None]
    F = np.cumsum(rho, axis=0) * dz - 0.5 * rho * dz
    n = P.shape[1]; F2 = F.reshape(F.shape[0], n * n)
    S = (F2.T @ F2).reshape(n, n, n, n) * dz * 1e-10
    Z, ov = q.kane_dipoles(z, P, E, V, r["m_edge"], r["Eg"], 0.0)
    return Z, S, ov


for NA in (5e19, 1e18):
    for ns in (3e13, 1e14):
        p = dict(BASE); p["NA"] = NA
        r = solve(ns, p)
        ZK, SK, ov = kane_ops(r)
        out = {}
        for tag in ("loewdin", "kane"):
            for kern in (False, True):
                k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=150,
                                            gamma=0.010, eps_b=eps_b(p, True), kernel=kern, zc=0.0, occ_tol=1e-5)
                if tag == "kane":
                    k.Z = ZK; k.S = SK
                L = np.array([k.loss(h) for h in HW]); c = np.cumsum(L) / L.sum()
                out[(tag, kern)] = float(np.interp(0.5, c, HW))
        print(f"NA {NA:.0e} ns {ns:.0e}: two-band overlap {ov:.1e} | median bare {out[('loewdin', False)]:.4f} -> "
              f"{out[('kane', False)]:.4f} | collective {out[('loewdin', True)]:.4f} -> {out[('kane', True)]:.4f} "
              f"({out[('kane', True)]/out[('loewdin', True)]-1:+.2%})", flush=True)
