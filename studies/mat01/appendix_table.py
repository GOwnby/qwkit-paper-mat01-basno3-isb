"""LaTeX rows for the model-check table of the MAT-01 appendix (from checks.json): change of E21 (meV) and of the
collective median (%) against the baseline, and the cap margin (eV), at 3e13 and 1e14 cm^-2 for both acceptor
densities."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(here, "checks.json")))
base = {(o["NA"], o["ns"]): o for o in C if o["var"] == "base"}
NAMES = [("basis2", "subbands to 2~eV"), ("basis4", "subbands to 4~eV"), ("L2", "box doubled"),
         ("xc_s", r"local XC, $\varepsilon_s$"), ("xc_inf", r"local XC, $\varepsilon_\infty$"),
         ("tdlda_inf", r"local XC + kernel, $\varepsilon_\infty$"), ("const_eps", r"no phonons ($\varepsilon_b=\varepsilon_\infty$)"),
         ("cap30", "3~nm cap"), ("fint_0.9", r"10\% of $\sigma_+$ on cap surface"), ("fint_0.75", r"25\% on cap surface"),
         ("fint_0.5", r"50\% on cap surface")]
cols = [(5e19, 3e13), (5e19, 1e14), (1e18, 3e13), (1e18, 1e14)]
print("baseline & " + " & ".join(f"{base[c]['E21']*1e3:.0f} / {base[c]['med_rpa']:.3f} / {base[c]['margin_cap']:+.2f}" for c in cols) + r" \\")
for v, name in NAMES:
    row = []
    for c in cols:
        o = [x for x in C if x["var"] == v and (x["NA"], x["ns"]) == c][0]
        b = base[c]
        flag = r"$^\ast$" if o["err"] > 1e-3 else ""
        dE = (o['E21'] - b['E21']) * 1e3; dm = 100 * (o['med_rpa'] / b['med_rpa'] - 1)
        fE = "0" if abs(dE) < 0.5 else f"{dE:+.0f}"; fm = "0.0" if abs(dm) < 0.05 else f"{dm:+.1f}"
        row.append(f"{fE} / {fm}\\% / {o['margin_cap']:+.2f}{flag}")
    print(name + " & " + " & ".join(row) + r" \\")
