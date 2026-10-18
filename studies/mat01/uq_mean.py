"""MAT-01 UQ supplement: the exact k = 0 mean absorption energies (meanmode.py) for the Latin-hypercube samples of
uq.json (same samples, n_s = 3e13 and 1e14 cm^-2): collective and single-particle mean, blue-shift factor, share of
the strongest mode; percentiles and Spearman rank correlations. Writes uq_mean.json."""
import json
from multiprocessing import Pool
from scipy.stats import spearmanr
from common import *
from uq import RANGES

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12


def means(r, p):
    z, psi, E = r["z"], r["psi"], r["E"]; dz = z[1] - z[0]
    nn = r["n_n"] * 1e4
    pairs = [(a, b) for a in range(len(E)) for b in range(a + 1, len(E)) if nn[a] - nn[b] > 1e-6 * nn.sum()]
    Ep = np.array([E[b] - E[a] for a, b in pairs]); Np = np.array([nn[a] - nn[b] for a, b in pairs])
    F = np.array([np.cumsum(psi[:, a] * psi[:, b]) * dz for a, b in pairs]).T
    K = (F.T @ F) * dz * 1e-10 * QE / (EPS0 * p["eps_inf"])
    Z = np.array([np.sum(psi[:, a] * z * psi[:, b]) * dz for a, b in pairs])
    s = np.sqrt(Ep * Np)
    w2, X = np.linalg.eigh(np.diag(Ep ** 2) + 2 * s[:, None] * K * s[None, :])
    Om = np.sqrt(np.clip(w2, 0, None))
    fb = Ep * Np * Z ** 2; fc = (X.T @ (s * Z)) ** 2
    return float(np.sum(fb * Ep) / fb.sum()), float(np.sum(fc * Om) / fc.sum()), float(fc.max() / fc.sum())


def job(args):
    i, p, ns = args
    try:
        r = solve(ns, p)
        eb, ec, top = means(r, p)
        return dict(i=i, ns=ns, ok=True, mean_bare=eb, mean_coll=ec, factor=ec / eb, top_share=top, E21=float(r["E"][1] - r["E"][0]))
    except Exception as e:
        return dict(i=i, ns=ns, ok=False, err=str(e))


if __name__ == "__main__":
    U = json.load(open(os.path.join(here, "uq.json")))
    P = U["samples"]
    jobs = [(i, P[i], ns) for ns in (3e13, 1e14) for i in range(len(P))]
    with Pool(1) as pool:
        res = pool.map(job, jobs, chunksize=4)
    out = dict(rows=res, stats={})
    for ns in (3e13, 1e14):
        rows = [r for r in res if r["ns"] == ns and r["ok"]]
        st = {}
        for q in ("mean_bare", "mean_coll", "factor", "top_share"):
            v = np.array([r[q] for r in rows]); pct = np.percentile(v, [5, 50, 95])
            rho = {k: float(spearmanr([P[r["i"]][k] for r in rows], v)[0]) for k in RANGES}
            st[q] = dict(p5=float(pct[0]), p50=float(pct[1]), p95=float(pct[2]), spearman=rho)
            top = sorted(rho.items(), key=lambda kv: -abs(kv[1]))[:3]
            print(f"ns {ns:.0e} {q:10s}: median {pct[1]:.4g} [{pct[0]:.4g}, {pct[2]:.4g}]  top: " + ", ".join(f"{k} {v:+.2f}" for k, v in top), flush=True)
        out["stats"][f"{ns:.0e}"] = st
    json.dump(out, open(os.path.join(here, "uq_mean.json"), "w"))
