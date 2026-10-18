"""MAT-01: electrostatic limit of electric-double-layer (top) gating through the LaInO3 cap. The interface carries a
fixed positive sheet sigma_P (polarization charge), chosen so that the ungated 2DEG has density n0; gate charge
added on the cap surface (sigma_g = n_s + N_dep - sigma_P) raises n_s and puts a field across the cap, which lowers
the cap conduction band towards the surface. For each n0 and cap variant the density is raised until 3% of the electrons sit in the cap;
reported: where the Fermi level reaches the lowest cap conduction-band energy (classical onset) and where 1% of the
electrons occupy cap-localized states (quantized onset). Variants: baseline cap (5 nm, eps 24), eps 50,
2 nm cap. The field-free cap of the main studies corresponds to all positive charge at the interface. Writes
gating.json."""
import json
from common import *

NS_GRID = [5e12, 7.5e12, 1e13, 1.25e13, 1.5e13, 1.75e13, 2e13, 2.25e13, 2.5e13, 2.75e13, 3e13, 3.5e13, 4e13, 4.5e13, 5e13,
           5.5e13, 6e13, 7e13, 8e13, 1e14, 1.2e14]
VARIANTS = {"base": dict(), "eps50": dict(eps_cap=50.0), "cap2nm": dict(d_cap=20.0)}


def run(NA, n0, var):
    p = dict(BASE); p["NA"] = NA
    p.update({k: v for k, v in VARIANTS[var].items() if k in p})
    kw = {"d_cap": VARIANTS[var]["d_cap"]} if "d_cap" in VARIANTS[var] else {}
    r0 = solve(n0, p, extend_to=0.0, **kw)
    sigP = n0 + r0["N_dep"]
    pts = []
    for ns in [x for x in NS_GRID if x >= n0]:
        r = solve(ns, p, extend_to=0.0, sigma_int=sigP, **kw)
        mc = float(np.min(r["V"][r["z"] < 0]) - r["mu"])
        dzz = r["z"][1] - r["z"][0]
        capw = np.array([np.sum(r["psi"][r["z"] < 0, j] ** 2) * dzz for j in range(len(r["E"]))])
        cf = float(np.sum(r["n_n"][capw > 0.5]) / ns)          # fraction of the electrons in cap-localized states
        pts.append(dict(ns=ns, margin_cap=mc, cap_frac=cf, E21=float(r["E"][1] - r["E"][0]), err=float(r["err"]),
                        sigma_gate=float(ns + r["N_dep"] - sigP)))
        print(f"  NA {NA:.0e} n0 {n0:.0e} {var}: ns {ns:.2e} gate {pts[-1]['sigma_gate']:.2e} margin_cap {mc:+.3f} cap fraction {cf:.4f} E21 {pts[-1]['E21']:.3f}", flush=True)
        if cf > 0.03:
            break
    m = np.array([q["margin_cap"] for q in pts]); x = np.array([q["ns"] for q in pts]); f = np.array([q["cap_frac"] for q in pts])
    ncross = float(np.interp(0.0, m[::-1], x[::-1])) if m[-1] < 0 <= m[0] else (float("nan") if m[0] < 0 else float("inf"))
    nq = float(np.interp(0.01, f, x)) if f[-1] > 0.01 > f[0] else float("inf")
    return dict(NA=NA, n0=n0, var=var, sigma_P=float(sigP), points=pts, ns_cross=ncross,
                dn_max=ncross - n0 if np.isfinite(ncross) else ncross, ns_cap1pct=nq, dn_cap1pct=nq - n0)


if __name__ == "__main__":
    out = []
    for NA in (5e19, 1e18):
        for n0 in (5e12, 1e13, 2e13):
            for var in VARIANTS:
                o = run(NA, n0, var); out.append(o)
                print(f"NA {NA:.0e} n0 {n0:.0e} {var}: cap band edge reached at n_s = {o['ns_cross']:.3g} (added {o['dn_max']:.3g}); "
                      f"1% of the electrons in the cap at {o['ns_cap1pct']:.3g} (added {o['dn_cap1pct']:.3g} cm^-2)", flush=True)
    json.dump(out, open(os.path.join(here, "gating.json"), "w"), indent=1)
