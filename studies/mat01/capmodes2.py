"""MAT-01 check: is the spreading of the collective strength at high density intrinsic, or caused by the quantized
LaInO3 cap states it overlaps? Compensated case at 6e13, 8e13 and 1e14 cm^-2 with the baseline offset (1.6 eV) and
with a 3.0 eV offset, which removes the cap states from the energy range of the modes. k = 0 modes (gamma = 2 meV)
and the k-resolved spectrum (gamma = 10 meV): median, strongest peak, f50. Writes capmodes2.json."""
import json
from common import *
from linear import modes, metrics

HW = np.arange(0.10, 2.5001, 0.005)
out = []
for ns in (6e13, 8e13, 1e14):
    for cbo in (1.6, 3.0):
        p = dict(BASE); p["NA"] = 5e19; p["cbo"] = cbo
        r = solve(ns, p)
        k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=200, gamma=0.010,
                                    eps_b=eps_b(p, True), kernel=True, zc=0.0, occ_tol=1e-5)
        L = np.array([k.loss(h) for h in HW]); m = metrics(HW, L); c = np.cumsum(L) / L.sum()
        mo = modes(r, p)
        o = dict(ns=ns, cbo=cbo, E21=float(r["E"][1] - r["E"][0]), median=float(np.interp(0.5, c, HW)), main=m["main"],
                 frac50=m["frac50"], modes=mo)
        out.append(o)
        print(f"ns {ns:.0e} CBO {cbo}: E21 {o['E21']:.3f} median {o['median']:.3f} peak {o['main']:.3f} f50 {o['frac50']:.2f} | modes "
              f"{np.round(mo['E'][:4], 3)} w {np.round(mo['weight'][:4], 2)}", flush=True)
json.dump(out, open(os.path.join(here, "capmodes2.json"), "w"), indent=1)
