"""MAT-01 convergence (after the basis fix): grid step, number of k shells, and the k = 0 (subband-edge) response
against the k-resolved one. n_s = 3e13 and 1e14 cm^-2, both acceptor densities, 300 K, gamma = 10 meV, phonon
background. Metrics over 0.10-2.5 eV: strongest peak, centroid, fraction within 50 meV of the peak (collective), and
the single-particle centroid. Writes conv2.json."""
import json
from common import *
from linear import metrics

HW = np.arange(0.10, 2.5001, 0.005)


def spectra(r, p, nk, k0=False):
    out = {}
    for tag, kern in (("bare", False), ("rpa", True)):
        if k0:
            k = krpa.KRPA(r["z"], r["psi"], r["E"][:, None], (r["n_n"] / r["n_n"].sum())[:, None],
                          np.array([r["n_n"].sum() * 1e4]), gamma=0.010, eps_b=eps_b(p, True), kernel=kern, zc=0.0,
                          occ_tol=1e-5)
        else:
            k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=nk, gamma=0.010,
                                        eps_b=eps_b(p, True), kernel=kern, zc=0.0, occ_tol=1e-5)
        L = np.array([k.loss(h) for h in HW]); m = metrics(HW, L); c = np.cumsum(L) / L.sum()
        out[tag] = dict(main=m["main"], centroid=m["centroid"], frac50=m["frac50"], median=float(np.interp(0.5, c, HW)))
    return out


rows = []
for NA in (5e19, 1e18):
    for ns in (3e13, 1e14):
        p = dict(BASE); p["NA"] = NA
        r = solve(ns, p)
        dz0 = r["z"][1] - r["z"][0]
        res = dict(NA=NA, ns=ns, dz=float(dz0), base=spectra(r, p, 200), nk400=spectra(r, p, 400), k0=spectra(r, p, 0, True))
        r2 = solve(ns, p, dz=dz0 / 2)
        res["dz_half"] = spectra(r2, p, 200)
        res["E21"] = float(r["E"][1] - r["E"][0]); res["E21_dz_half"] = float(r2["E"][1] - r2["E"][0])
        rows.append(res)
        print(f"NA {NA:.0e} ns {ns:.0e} dz {dz0}: E21 {res['E21']:.4f} (dz/2 {res['E21_dz_half']:.4f}) | " + " | ".join(
            f"{v}: peak {res[v]['rpa']['main']:.3f} med {res[v]['rpa']['median']:.4f} cen {res[v]['rpa']['centroid']:.4f} f50 {res[v]['rpa']['frac50']:.2f} bare med {res[v]['bare']['median']:.4f}"
            for v in ("base", "nk400", "dz_half", "k0")), flush=True)
json.dump(rows, open(os.path.join(here, "conv2.json"), "w"), indent=1)
