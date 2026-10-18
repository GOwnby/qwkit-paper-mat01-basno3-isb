"""MAT-01 check: dependence of the collective mode structure on the LaInO3 cap (whose hard-walled states lie
0.7-0.8 eV above E_1 at 1e14 cm^-2 in the compensated case). Caps of 3, 5 and 8 nm at n_s = 3e13 and 1e14 cm^-2,
both acceptor densities: k = 0 modes (gamma = 2 meV) and their shares, and the k-resolved collective spectrum
(gamma = 10 meV): median, strongest peak, fraction within 50 meV of it. Also the cap weight of the final states of
the strongest 1 -> n transitions. Writes capmodes.json."""
import json
from common import *
from linear import modes, metrics

HW = np.arange(0.10, 2.5001, 0.005)
out = []
for NA in (5e19, 1e18):
    for ns in (3e13, 1e14):
        for d in (30.0, 50.0, 80.0):
            p = dict(BASE); p["NA"] = NA
            r = solve(ns, p, d_cap=d)
            k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=200, gamma=0.010,
                                        eps_b=eps_b(p, True), kernel=True, zc=0.0, occ_tol=1e-5)
            L = np.array([k.loss(h) for h in HW]); m = metrics(HW, L)
            c = np.cumsum(L) / L.sum(); med = float(np.interp(0.5, c, HW))
            mo = modes(r, p)
            dzz = r["z"][1] - r["z"][0]
            Z = (r["psi"].T * r["z"]) @ r["psi"] * dzz
            f = np.array([(r["E"][j] - r["E"][0]) * Z[0, j] ** 2 for j in range(len(r["E"]))]); f[0] = 0
            capw = np.array([np.sum(r["psi"][r["z"] < 0, j] ** 2) * dzz for j in range(len(r["E"]))])
            fcap = float(np.sum(f[capw > 0.5]) / f.sum())
            o = dict(NA=NA, ns=ns, d_cap=d, median=med, main=m["main"], frac50=m["frac50"], modes=mo, f_cap_states=fcap)
            out.append(o)
            print(f"NA {NA:.0e} ns {ns:.0e} cap {d/10:.0f} nm: median {med:.3f} peak {m['main']:.3f} f50 {m['frac50']:.2f} | modes "
                  f"{np.round(mo['E'][:4], 3)} w {np.round(mo['weight'][:4], 2)} | 1->n strength to cap states {fcap:.3f}", flush=True)
json.dump(out, open(os.path.join(here, "capmodes.json"), "w"), indent=1)
