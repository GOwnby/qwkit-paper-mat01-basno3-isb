"""MAT-01 checks (revised), at n_s = 3e13 and 1e14 cm^-2, N_A = 5e19 and 1e18 cm^-3, 300 K. For each variant: E21,
E_F - E1, the single-particle and collective (RPA) centroids over 0.10-2.5 eV, and the margin between the Fermi level
and the lowest LaInO3 conduction-band energy in the cap.
  base       baseline (field-free cap, subbands up to 3 eV above E1, box length rule)
  basis2/4   subbands up to 2 and 4 eV above E1 (at most 90)
  L2         box twice as long
  xc_s/xc_inf  static local exchange-correlation with eps_s / eps_inf in the self-consistent potential
  tdlda_inf  xc_inf plus the adiabatic local XC kernel in the response (eps_inf)
  const_eps  constant eps_inf background (no phonons) in the response
  cap30      3 nm cap
  fint_x     fraction x of the positive sheet at the interface, the rest on the cap surface (gate charge
             unscreened in the cap): x = 0.9, 0.75, 0.5
Writes checks.json."""
import json
from multiprocessing import Pool
from common import *

HW = np.arange(0.10, 2.5001, 0.01)


def cen(r, p, kernel=True, phonons=True, fxc=None):
    k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=150, gamma=p["gamma"],
                                eps_b=eps_b(p, phonons), kernel=kernel, zc=0.0, occ_tol=1e-5)
    if fxc is not None:
        k.set_xc(r["z"], r["psi"], fxc)
    L = np.array([k.loss(h) for h in HW])
    c = np.cumsum(L) / np.sum(L)
    return float(np.sum(HW * L) / np.sum(L)), float(np.interp(0.5, c, HW))


def summarize(r, p, **kw):
    z = r["z"]
    capmin = float(np.min(r["V"][z < 0]) - r["mu"])
    cb, mb = cen(r, p, kernel=False); cr, mr = cen(r, p, **kw)
    return dict(E21=float(r["E"][1] - r["E"][0]), EF=float(r["mu"] - r["E"][0]), cen_bare=cb, cen_rpa=cr, med_bare=mb,
                med_rpa=mr, margin_cap=capmin, n_states=len(r["E"]))


def job(args):
    NA, ns, var = args
    p = dict(BASE); p["NA"] = NA
    kw = {}
    if var == "base":
        r = solve(ns, p)
    elif var in ("basis2", "basis4"):
        r = solve(ns, p, extend_to=2.0 if var == "basis2" else 4.0)
    elif var == "L2":
        r = solve(ns, p, Lscale=2.0)
    elif var == "xc_s":
        r = solve(ns, p, xc=True)
    elif var in ("xc_inf", "tdlda_inf"):
        r = solve(ns, p, xc=True, xc_eps=p["eps_inf"])
        if var == "tdlda_inf":
            n3 = (r["psi"] ** 2) @ r["n_n"] * 1e8
            kw["fxc"] = I.xc_kernel(n3, p["eps_inf"], p["m"]) * (r["z"] > 0)
    elif var == "const_eps":
        r = solve(ns, p); kw["phonons"] = False
    elif var == "cap30":
        r = solve(ns, p, d_cap=30.0)
    elif var.startswith("fint_"):
        q = dict(p); q["f_int"] = float(var.split("_")[1]); p = q
        r = solve(ns, p)
    out = dict(NA=NA, ns=ns, var=var, err=float(r["err"]))
    out.update(summarize(r, p, **kw))
    return out


if __name__ == "__main__":
    VARS = ["base", "basis2", "basis4", "L2", "xc_s", "xc_inf", "tdlda_inf", "const_eps", "cap30", "fint_0.9", "fint_0.75", "fint_0.5"]
    jobs = [(NA, ns, v) for NA in (5e19, 1e18) for ns in (3e13, 1e14) for v in VARS]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    for o in res:
        print(f"NA {o['NA']:.0e} ns {o['ns']:.0e} {o['var']:10s}: n {o['n_states']:2d} E21 {o['E21']:.3f} EF {o['EF']:.3f} cen bare {o['cen_bare']:.3f} "
              f"rpa {o['cen_rpa']:.3f} | median bare {o['med_bare']:.3f} rpa {o['med_rpa']:.3f} | margin(cap) {o['margin_cap']:+.3f} err {o['err']:.0e}", flush=True)
    json.dump(res, open(os.path.join(here, "checks.json"), "w"), indent=1)
