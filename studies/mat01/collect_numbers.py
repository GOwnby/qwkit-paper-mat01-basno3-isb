"""Every number quoted in the MAT-01 manuscript, from the study outputs (linear, gating, capmodes, extras logs,
lifetimes, chi2, checks, conv2, fsum, kane_check, kane_rpa, basis_clean, uq). Prints a flat list; the manuscript is
checked against it."""
import json, os, re
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(here, f)))
HC = 1.23984198                                       # eV um
NAME = {5e19: "compensated", 1e18: "clean"}
rng = lambda v, f="{:.3g}": f"{f.format(min(v))}-{f.format(max(v))}"

D = J("linear.json")
for NA in (5e19, 1e18):
    R = sorted([r for r in D["rows"] if r["NA"] == NA and r["T"] == 300.0], key=lambda r: r["ns"])
    n = NAME[NA]; ns = np.array([r["ns"] for r in R])
    g = lambda f: [f(r) for r in R]
    print(f"== {n} (N_A {NA:.0e}), n_s {ns[0]:.1e}-{ns[-1]:.1e}")
    print("  E21", rng(g(lambda r: r["E_rel"][1]), "{:.3f}"), "| E_F-E1", rng(g(lambda r: r["mu_rel"]), "{:.3f}"),
          "| <z>1 (A)", rng(g(lambda r: r["z1"]), "{:.1f}"), "| m1", rng(g(lambda r: r["m_n"][0]), "{:.3f}"),
          "| m2/m1", rng(g(lambda r: r["m_n"][1] / r["m_n"][0]), "{:.3f}"))
    print("  N_dep", rng(g(lambda r: r["N_dep"]), "{:.2e}"), "| w_d (A)", rng(g(lambda r: r["w_d"]), "{:.0f}"),
          "| occ2", [round(r["occ"][1], 3) for r in R], "| max residual", max(g(lambda r: r["err"])),
          "| basis reaches", rng(g(lambda r: r["n_basis_emax"]), "{:.2f}"), "eV")
    mc = np.array(g(lambda r: r["margin_cap"]))
    cr = lambda t: float(np.interp(t, mc[::-1], ns[::-1])) if mc.min() < t < mc.max() else None
    print("  margin_cap", [round(x, 3) for x in mc], "| 0.2 eV at", cr(0.2), "| 0 at", cr(0.0))
    caps = [(r["ns"], j, round(r["E_rel"][j], 3), round(r["capw"][j], 2)) for r in R for j in range(1, 8) if r["capw"][j] >= 0.5]
    print("  cap states (ns, index, E-E1, cap weight):", caps[:12])
    for tag in ("bare", "rpa", "rpa30"):
        print(f"  {tag}: median", rng(g(lambda r: r[tag]["median"]), "{:.3f}"), "| peak", rng(g(lambda r: r[tag]["main"]), "{:.3f}"),
              "| f50", [round(r[tag]["frac50"], 2) for r in R], "| fwhm", rng(g(lambda r: r[tag]["fwhm"]), "{:.3f}"),
              "| centroid", rng(g(lambda r: r[tag]["centroid"]), "{:.3f}"))
    med = g(lambda r: r["rpa"]["median"]); pk = g(lambda r: r["rpa"]["main"])
    print("  collective median wavelengths (um)", rng([HC / x for x in med], "{:.2f}"), "| strongest-peak wavelengths", rng([HC / x for x in pk], "{:.2f}"))
    print("  median ratio coll/single", rng(g(lambda r: r["ratio_med_rpa_bare"]), "{:.2f}"), "| coll median/E21", rng(g(lambda r: r["rpa"]["median"] / r["E_rel"][1]), "{:.2f}"),
          "| single median/E21", rng(g(lambda r: r["bare"]["median"] / r["E_rel"][1]), "{:.3f}"))
    print("  interquartile width coll (eV)", [round(r["rpa"]["quartiles"][1] - r["rpa"]["quartiles"][0], 3) for r in R])
    print("  gamma30 vs 10 median shift", rng(g(lambda r: r["rpa30"]["median"] / r["rpa"]["median"] - 1), "{:+.3f}"))
    for r in R:
        if "modes" in r:
            print(f"  modes {r['ns']:.0e}:", np.round(r["modes"]["E"], 3).tolist(), np.round(r["modes"]["weight"], 2).tolist())
    T10 = {r["ns"]: r for r in D["rows"] if r["NA"] == NA and r["T"] == 10.0}
    d10 = [abs(T10[s]["rpa"]["median"] / r["rpa"]["median"] - 1) for s in T10 for r in R if r["ns"] == s]
    print("  10 K vs 300 K collective median: max", round(max(d10), 4), "| occ2 at 10 K", {f"{s:.0e}": round(T10[s]["occ"][1], 3) for s in T10})
    print("  z12 range", rng(g(lambda r: r["z1n"][0]), "{:.2f}"), "| z13 max", round(max(g(lambda r: r["z1n"][1])), 2))

def show(f):
    p = os.path.join(here, f)
    if os.path.exists(p):
        print(f"== {f}")
        for line in open(p):
            if not re.search("warn", line, re.I) and not line.startswith("  "):
                print("  " + line.rstrip()[:400])
for f in ("gating.log", "meanmode.log", "capmodes2.log", "extras.log", "capmodes.log", "conv2.log", "fsum.log", "kane_check.log", "kane_rpa.log",
          "basis_clean.log", "lifetimes.log", "chi2.log", "checks.log"):
    show(f)
M = J("meanmode.json")
for NA in (5e19, 1e18):
    for lim in (1e14, 1.2e14):
        R = [o for o in M if o["NA"] == NA and o["ns"] <= lim]
        print(f"== meanmode {NAME[NA]} n_s <= {lim:.1e}: mean bare", rng([o['mean_bare'] for o in R], "{:.3f}"), "| mean coll", rng([o['mean_coll'] for o in R], "{:.3f}"),
              "| factor", rng([o['ratio'] for o in R], "{:.2f}"), "| mean/E21", rng([o['mean_coll'] / o['E21'] for o in R], "{:.2f}"),
              "| single mean/E21", rng([o['mean_bare'] / o['E21'] for o in R], "{:.2f}"), "| dressed 1->2", rng([o['E12_dressed'] for o in R], "{:.3f}"),
              "| top share", [round(o['top_share'], 2) for o in R])
if os.path.exists(os.path.join(here, "uq.json")):
    U = J("uq.json")
    print("== uq")
    for k, st in U["stats"].items():
        print(" ", k, "n_ok", st["n_ok"], "frac margin<0.2", st["frac_margin_lt_0.2"], "frac margin<0", st.get("frac_margin_lt_0"))
        for q in ("E21", "median", "median_bare", "ratio_med", "peak", "chi_s", "tau21_fs", "margin_cap"):
            if q in st:
                s = st[q]; top = sorted(s["spearman"].items(), key=lambda kv: -abs(kv[1]))[:3]
                print(f"    {q}: p5 {s['p5']:.4g} p50 {s['p50']:.4g} p95 {s['p95']:.4g} | " + ", ".join(f"{a} {b:+.2f}" for a, b in top))
if os.path.exists(os.path.join(here, "radiative.json")):
    print("== radiative (45 deg inside BSO): strongest-mode hbar Gamma_rad max (meV)",
          round(max(o["hG_top_45"] for o in J("radiative.json")) * 1e3, 2), "| all weight in one mode max",
          round(max(o["hG_all_45"] for o in J("radiative.json")) * 1e3, 2), "| peak absorption max (gamma 10, 30 meV)",
          round(max(o["A_top_45"]["10"] for o in J("radiative.json")), 3), round(max(o["A_top_45"]["30"] for o in J("radiative.json")), 3))
if os.path.exists(os.path.join(here, "gating.json")):
    for o in J("gating.json"):
        if o["n0"] == 5e12:
            print(f"== gating n0 = 5e12 {NAME[o['NA']]} {o['var']}: classical edge n_s {o['ns_cross']:.2e}, 1% cap onset n_s {o['ns_cap1pct']:.2e}")
