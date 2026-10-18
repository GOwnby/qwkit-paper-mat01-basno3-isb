"""MAT-01 study 4 (revised): second-harmonic susceptibility of the LaInO3/BaSnO3 2DEG.
Sheet susceptibility chi_s = P2_sheet/(eps0 E_d^2) (nm^2/V), single particle and k-resolved collective (second-order
RPA, phonon background at w and 2w), pump 0.10-1.2 eV, n_s = 1e13 ... 1e14 cm^-2, N_A = 5e19 and 1e18 cm^-3, 300 K,
linewidths 10 and 30 meV. Reference: a doubly resonant InGaAs/InAlAs asymmetric double well (100 A
barriers, 5e11 cm^-2 per period, self-consistent, one shell) with the same linewidths, as a sheet per period.
Reference structure: 70/12/32 A (E31/E21 = 1.97 self-consistently, doubly resonant within 3 meV), chosen by a scan.
Peaks refined on a 1 meV grid. Writes chi2.json."""
import json
from multiprocessing import Pool
from common import *
from qwkit import sp

HW = np.arange(0.10, 1.2001, 0.01)
NSC = [1e13, 2e13, 3e13, 5e13, 7e13, 1e14]
EPS0_ = 8.8541878128e-12


def peak(k, c):
    """Refine the coarse spectrum c (on HW): 1 meV grid (+-12 meV) around its three largest local maxima. The 0.01 eV
    grid under-samples the gamma = 10 meV double resonances (width in pump energy about gamma/2)."""
    loc = [j for j in range(len(c)) if (j == 0 or c[j] >= c[j - 1]) and (j == len(c) - 1 or c[j] >= c[j + 1])]
    best = (float(c.max()), float(HW[int(np.argmax(c))]))
    for j in sorted(loc, key=lambda j: -c[j])[:3]:
        hf = np.arange(max(HW[j] - 0.012, HW[0]), HW[j] + 0.0121, 0.001)
        cf = np.abs(np.array([k.sheet_p2(h) for h in hf])) / EPS0_ * 1e18
        if cf.max() > best[0]:
            best = (float(cf.max()), float(hf[int(np.argmax(cf))]))
    return best


def job(args):
    NA, ns = args
    p = dict(BASE); p["NA"] = NA
    r = solve(ns, p)
    out = dict(NA=NA, ns=ns, n_states=len(r["E"]), E_rel=[float(e - r["E"][0]) for e in r["E"][:6]])
    for g in (0.010, 0.030):
        for tag, kern in (("bare", False), ("rpa", True)):
            k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=150, gamma=g,
                                        eps_b=eps_b(p, True), kernel=kern, zc=0.0, occ_tol=1e-5)
            c = np.abs(np.array([k.sheet_p2(h) for h in HW])) / EPS0_ * 1e18
            pk, hw = peak(k, c)
            out[f"{tag}_{g*1e3:.0f}"] = dict(peak_nm2V=pk, hw=hw, peak_coarse=float(c.max()), abs=c.tolist())
    return out


def reference():
    lay = [(70.0, "well"), (12.0, "barrier"), (32.0, "well")]
    r = sp.self_consistent(lay, 0.0, ns_cm2=5e11, T=300.0, spacer=100.0, dz=0.5, doping="spacer", nstates=6)
    E, psi, Z, rho = sp.prepare(r, nkeep=4)
    z = r["z"]; zc = 0.5 * (r["active"][0] + r["active"][1])
    out = dict(E21=float(E[1] - E[0]), E31=float(E[2] - E[0]), period_A=214.0, layers="70/12/32 A")
    hw = np.arange(0.03, 0.20, 0.0005)
    for g in (0.010, 0.030):
        k = krpa.KRPA(z, psi, E[:, None], rho[:, None], np.array([5e11 * 1e4]), gamma=g, eps_b=10.9, kernel=True, zc=zc)
        c = np.array([abs(k.sheet_p2(h)) for h in hw]) / EPS0_ * 1e18
        i = int(np.argmax(c))
        out[f"g{g*1e3:.0f}"] = dict(peak_sheet_nm2V=float(c[i]), hw=float(hw[i]), bulk_pmV=float(c[i] / (214.0 * 0.1) * 1e3))
    return out


if __name__ == "__main__":
    ref = reference()
    print("reference ADQW 70/12/32 A (per period):", ref, flush=True)
    jobs = [(NA, ns) for NA in (5e19, 1e18) for ns in NSC]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    for o in res:
        print(f"NA {o['NA']:.0e} ns {o['ns']:.0e}: E21 {o['E_rel'][1]:.3f} E31 {o['E_rel'][2]:.3f} | g10 bare {o['bare_10']['peak_nm2V']:.0f} at {o['bare_10']['hw']:.3f}"
              f" rpa {o['rpa_10']['peak_nm2V']:.0f} at {o['rpa_10']['hw']:.3f} | g30 bare {o['bare_30']['peak_nm2V']:.0f} rpa {o['rpa_30']['peak_nm2V']:.0f} at {o['rpa_30']['hw']:.3f}"
              f" | rpa/ref g10 {o['rpa_10']['peak_nm2V']/ref['g10']['peak_sheet_nm2V']:.2f} g30 {o['rpa_30']['peak_nm2V']/ref['g30']['peak_sheet_nm2V']:.2f}", flush=True)
    json.dump(dict(hw=HW.tolist(), reference=ref, rows=res), open(os.path.join(here, "chi2.json"), "w"))
