"""MAT-01 study 3 (revised): LO-phonon intersubband relaxation 2 -> 1 of the LaInO3/BaSnO3 2DEG at 300 K.
Three polar modes of BaSnO3 with per-mode Froehlich weights (residues of the factorized 1/eps), bulk phonons.
For an electron excited vertically from subband 1 at in-plane wavevector k into subband 2 (the states an
intersubband absorption creates), the rate into subband 1 includes the angular integral over the final wavevector,
the nonparabolic final density of states and Pauli blocking (1 - f_1); it is averaged over the excitation
distribution f_1(k)(1 - f_2(k)) k. Screening: static RPA by the intrasubband polarization of the occupied subbands
with Coulomb form factors (scattering.screening_factor), eps_inf background; the earlier edge-electron, unblocked,
unscreened rate is kept for comparison. Writes lifetimes.json."""
import json
from multiprocessing import Pool
from scipy.interpolate import interp1d
from common import *

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; HBAR = 1.054571817e-34; M0 = 9.1093837015e-31
NSL = [5e12, 1e13, 2e13, 3e13, 5e13, 7e13, 1e14]


def job(args):
    NA, ns = args
    p = dict(BASE); p["NA"] = NA
    out = job_params(p, ns, full=True)
    out.update(NA=NA, ns=ns)
    return out


def job_params(p, ns, full=False):
    r = solve(ns, p, extend_to=0.0)                         # only the subbands of the self-consistent set are needed
    z, psi, E, m_n, G_n, mu, T = r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"]
    C = krpa.froehlich_weights(p["eps_inf"], MODES)
    modes = [(l, c) for (t, l), c in zip(MODES, C)]
    phi = psi[:, 1] * psi[:, 0]
    Qg = np.logspace(7, 10.7, 70)
    IQ0 = np.array([sc.form_integral(z, phi, q) / q for q in Qg])
    occ = [(l, m_n[l], np.sqrt(2 * np.pi * r["n_n"][l] * 1e4)) for l in range(len(E)) if r["n_n"][l] / ns > 0.01]
    Sg = sc.screening_factor(z, psi, 1, 0, occ, Qg, p["eps_inf"])
    IQ = interp1d(np.log(Qg), IQ0, kind="cubic", fill_value="extrapolate")
    IQs = interp1d(np.log(Qg), IQ0 * Sg, kind="cubic", fill_value="extrapolate")
    fI = lambda Q: IQ(np.log(np.clip(Q, Qg[0], Qg[-1])))
    fIs = lambda Q: IQs(np.log(np.clip(Q, Qg[0], Qg[-1])))
    # excitation distribution: vertical 1 -> 2 at the same k
    kT = 8.617333262e-5 * T
    x0 = np.linspace(1e-6, m_n[0] * (max(mu - E[0], 0) + 8 * kT) * (1 + (max(mu - E[0], 0) + 8 * kT) / G_n[0]), 40)
    eps1 = I.dispersion(x0 / m_n[0], m_n[0], G_n[0]); eps2 = I.dispersion(x0 / m_n[1], m_n[1], G_n[1])
    f1 = 1 / (1 + np.exp((E[0] + eps1 - mu) / kT)); f2 = 1 / (1 + np.exp((E[1] + eps2 - mu) / kT))
    wts = f1 * (1 - f2)
    kk = np.sqrt(2 * M0 * x0 * QE) / HBAR
    res = {}
    tags = (("pauli_unscreened", mu, fI), ("pauli_screened", mu, fIs), ("unblocked_unscreened", None, fI)) if full else (("pauli_screened", mu, fIs),)
    for tag, mu_, fun in tags:
        W = np.array([sc.lo_rate_k(z, psi[:, 1], psi[:, 0], k, E[1] + e2, E[0], m_n[0], G_n[0], modes, T=T, mu=mu_, IQ=fun)
                      for k, e2 in zip(kk, eps2)])
        res[tag] = float(1e15 / (np.sum(wts * W) / np.sum(wts)))          # fs
    if not full:
        return res
    edge = float(1e15 / sc.lo_rate_modes(z, psi[:, 1], psi[:, 0], E[1] - E[0], modes, T=T, m_eff=m_n[0]))
    blocked = 1 - 1 / (1 + np.exp((E[1] - modes[1][0] - mu) / kT))
    return dict(E21=float(E[1] - E[0]), EF_rel=float(mu - E[0]), tau_fs=res, tau_edge_unscreened_fs=edge,
                S_at_Q=dict(Q=Qg.tolist(), S=Sg.tolist()), mean_S=float(np.interp(np.log(np.sqrt(2 * m_n[0] * M0 * (E[1] - E[0] - modes[1][0]) * QE) / HBAR), np.log(Qg), Sg)) if E[1] - E[0] > modes[1][0] else None,
                n_occupied=len(occ))


if __name__ == "__main__":
    C = krpa.froehlich_weights(BASE["eps_inf"], MODES)
    alpha = [QE ** 2 / (4 * np.pi * EPS0 * HBAR) * np.sqrt(0.19 * M0 / (2 * l * QE)) * c for (t, l), c in zip(MODES, C)]
    print("Froehlich weights", np.round(C, 4), "alpha per mode (m 0.19)", np.round(alpha, 3), "sum", round(sum(alpha), 2), flush=True)
    jobs = [(NA, ns) for NA in (5e19, 1e18) for ns in NSL]
    with Pool(2) as pool:
        res = pool.map(job, jobs)
    for o in res:
        t = o["tau_fs"]
        print(f"NA {o['NA']:.0e} ns {o['ns']:.0e}: E21 {o['E21']:.3f} EF-E1 {o['EF_rel']:.3f} | edge (old) {o['tau_edge_unscreened_fs']:.0f} fs | "
              f"k-avg unblocked {t['unblocked_unscreened']:.0f} | Pauli {t['pauli_unscreened']:.0f} | Pauli+screened {t['pauli_screened']:.0f} fs | "
              f"S(Q_emit) {o['mean_S']}", flush=True)
    json.dump(dict(weights=C.tolist(), alpha=alpha, rows=res), open(os.path.join(here, "lifetimes.json"), "w"))
