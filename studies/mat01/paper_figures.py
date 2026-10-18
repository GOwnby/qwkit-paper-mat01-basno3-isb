"""Figures for the MAT-01 manuscript from linear.json, lifetimes.json, chi2.json and uq.json.
Writes ../../paper_mat01/figures/*.pdf."""
import os, json, sys
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import BASE, solve

outdir = os.path.join(here, "..", "..", "paper_mat01", "figures"); os.makedirs(outdir, exist_ok=True)
BLUE, ORANGE, AQUA, PURPLE, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a4fbf", "#c4314b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 6.6, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 1.3, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
J = lambda f: json.load(open(os.path.join(here, f)))
NAC = {5e19: (BLUE, "-", r"$N_A=5\times10^{19}$"), 1e18: (ORANGE, "--", r"$N_A=10^{18}$")}


def clean(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def label(ax, s, x=-0.2):
    ax.text(x, 1.03, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")


def rows(d, NA, T=300.0):
    return sorted([r for r in d["rows"] if r["NA"] == NA and r["T"] == T], key=lambda r: r["ns"])


def fig1():
    D = J("linear.json")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.45, width_ratios=[1.15, 1, 1]))
    ax = axs[0]
    for ns, col in ((3e13, BLUE), (1e14, PURPLE)):
        r = solve(ns, BASE, extend_to=0.0)
        z, V = r["z"], r["V"]; sel = (z > -55) & (z < 130)
        ax.plot(z[sel], V[sel], color=col, lw=1.1, label=(r"$n_s=3\times10^{13}$" if ns < 5e13 else r"$n_s=10^{14}$ cm$^{-2}$"))
        for n in range(2):
            w = r["psi"][:, n] ** 2
            ax.plot(z[sel], r["E"][n] + 0.9 * w[sel] / w.max() * 0.12, color=col, lw=0.7, alpha=0.8)
        ax.axhline(r["mu"], color=col, lw=0.7, ls=":")
    ax.set_xlabel("$z$ (Å)"); ax.set_ylabel("energy (eV)"); ax.set_ylim(-0.1, 2.0)
    ax.text(-52, 1.80, "LaInO$_3$ cap", fontsize=6.0, color=INK2); ax.text(60, 0.45, "BaSnO$_3$", fontsize=6.5, color=INK2)
    ax.legend(frameon=False, loc="lower right", fontsize=6.2); clean(ax); label(ax, "a", -0.25)
    ax = axs[1]
    for NA, (col, ls, lab) in NAC.items():
        R = rows(D, NA); ns = np.array([r["ns"] for r in R]) / 1e13
        ax.plot(ns, [r["E_rel"][1] for r in R], color=col, ls=ls, lw=1.3, label=lab)
        # third BaSnO3 subband (states with more than half their weight in the cap are shown separately)
        e3 = [next(r["E_rel"][j] for j in range(2, 8) if r["capw"][j] < 0.5) for r in R]
        ax.plot(ns, e3, color=col, ls=ls, lw=0.8)
        cs = [(x, r["E_rel"][j]) for x, r in zip(ns, R) for j in range(1, 8) if r["capw"][j] >= 0.5 and r["E_rel"][j] < 1.1]
        if cs:
            ax.plot([c[0] for c in cs], [c[1] for c in cs], ls="none", marker="x", ms=3, color=col, mew=0.8)
        ax.plot(ns, [r["mu_rel"] for r in R], color=col, ls=ls, lw=0.9, marker="o", ms=2.2)
    ax.set_xlabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)"); ax.set_ylabel(r"$E_n-E_1$, $E_F-E_1$ (eV)")
    ax.text(12.4, 0.70, r"$E_F$", fontsize=6.0); ax.text(12.4, 0.60, r"$E_2$", fontsize=6.0, color=NAC[5e19][0])
    ax.text(12.4, 0.43, r"$E_2$", fontsize=6.0, color=NAC[1e18][0])
    ax.set_ylim(0, 1.15); ax.set_xlim(0, 13.6)
    ax.legend(frameon=False, loc="lower right", fontsize=5.6, bbox_to_anchor=(1.0, 0.0)); clean(ax); label(ax, "b", -0.27)
    ax = axs[2]
    for NA, (col, ls, lab) in NAC.items():
        R = rows(D, NA); ns = np.array([r["ns"] for r in R]) / 1e13
        ax.plot(ns, [r["margin_cap"] for r in R], color=col, ls=ls, label=lab)
        ax2 = ax
    ax.axhline(0, color=INK2, lw=0.6); ax.axhline(0.2, color=INK2, lw=0.5, ls=":")
    ax.set_xlabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)"); ax.set_ylabel(r"$E_c^{\rm LaInO_3}-E_F$ (eV)")
    ax.legend(frameon=False, fontsize=6.0); clean(ax); label(ax, "c", -0.27)
    fig.savefig(os.path.join(outdir, "fig1_subbands.pdf")); plt.close(fig)


def fig2():
    D = J("linear.json"); hw = np.array(D["hw"])
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True, gridspec_kw=dict(hspace=0.12, wspace=0.12))
    for j, NA in enumerate((5e19, 1e18)):
        R = rows(D, NA); ns = np.array([r["ns"] for r in R]) / 1e13
        for i, tag in enumerate(("bare", "rpa")):
            ax = axs[i, j]
            M = np.array([np.array(r["spectra"][tag]) / max(np.array(r["spectra"][tag])[hw >= 0.15]) for r in R])
            edges = np.concatenate([[ns[0] - 0.25], 0.5 * (ns[1:] + ns[:-1]), [ns[-1] + 1.0]])
            hwe = np.concatenate([[hw[0] - 0.00125], 0.5 * (hw[1:] + hw[:-1]), [hw[-1] + 0.00125]])
            im = ax.pcolormesh(hwe, edges, np.clip(M, 0, 1), cmap="magma", shading="flat", rasterized=True)
            ax.plot([r["E_rel"][1] for r in R], ns, color="white", ls="--", lw=0.8)
            ax.plot([r[tag]["median"] for r in R], ns, color=AQUA, marker="o", ms=2.2, lw=0.8)
            ax.set_xlim(0.1, 1.8)
            if j == 0:
                ax.set_ylabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)")
            else:
                ax.set_yticklabels([])
            if i == 1:
                ax.set_xlabel("photon energy (eV)")
            ttl = ("single particle" if tag == "bare" else "collective (k-resolved RPA)") + "\n" + NAC[NA][2] + r" cm$^{-3}$"
            ax.text(0.98, 0.04, ttl, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color="white")
            label(ax, "abcd"[2 * i + j], -0.12)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.015); cb.set_label("absorbed power (normalized per row)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    fig.savefig(os.path.join(outdir, "fig2_spectra.pdf")); plt.close(fig)


def fig3():
    D = J("linear.json"); hw = np.array(D["hw"])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.42))
    ax = axs[0]
    MM = J("meanmode.json")
    for NA, (col, ls, lab) in NAC.items():
        R = sorted([o for o in MM if o["NA"] == NA], key=lambda o: o["ns"]); ns = np.array([o["ns"] for o in R]) / 1e13
        ax.plot(ns, [o["E21"] for o in R], color=col, ls=":", lw=1.0)
        ax.plot(ns, [o["mean_bare"] for o in R], color=col, ls="none", marker="o", ms=2.4, mfc="white")
        ax.plot(ns, [o["mean_coll"] for o in R], color=col, ls=ls, label=lab)
    from matplotlib.lines import Line2D
    h = ax.get_legend_handles_labels()[0] + [Line2D([], [], color=INK2, ls="-", lw=1.0), Line2D([], [], color=INK2, ls="none", marker="o", ms=2.4, mfc="white"),
                                             Line2D([], [], color=INK2, ls=":", lw=1.0)]
    l = ax.get_legend_handles_labels()[1] + [r"collective $\bar E$", r"single-particle $\bar E$", r"$E_{21}$"]
    ax.set_xlabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)"); ax.set_ylabel("energy (eV)")
    ax.legend(h, l, frameon=False, fontsize=5.6, loc="upper left"); clean(ax); label(ax, "a", -0.24)

    ax = axs[1]
    for NA, (col, ls, lab) in NAC.items():
        for o in MM:
            if o["NA"] == NA and o["ns"] in (1e13, 3e13, 1e14):
                x = o["ns"] / 1e13 + (0.15 if NA == 1e18 else -0.15)
                for e, w in zip(o["modes_E"], o["modes_w"]):
                    if w > 0.03:
                        ax.scatter([x], [e], s=120 * w, color=col, alpha=0.7, lw=0)

    ax.set_xlabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)"); ax.set_ylabel("collective mode energy (eV)")
    ax.set_xticks([1, 3, 10]); ax.set_xlim(0, 11)
    ax.text(0.03, 0.97, "marker area: share of absorption (exact k = 0 modes)\nblue: compensated, orange: clean", transform=ax.transAxes, fontsize=5.4, color=INK2, va="top")
    ax.set_ylim(0.1, 1.5)
    clean(ax); label(ax, "b", -0.27)
    ax = axs[2]
    for NA, (col, ls, lab) in NAC.items():
        r = [x for x in rows(D, NA) if x["ns"] == 1e14][0]
        ax.plot(hw, r["spectra"]["rpa"], color=col, ls=ls, lw=0.9, label=lab + r", $\gamma$=10 meV")
        ax.plot(hw, np.array(r["spectra"]["rpa30"]), color=col, ls=ls, lw=1.8, alpha=0.45, label=lab + r", $\gamma$=30 meV")
    ax.set_xlim(0.1, 1.8); ax.set_xlabel("photon energy (eV)"); ax.set_ylabel("absorbed power (norm.)")
    ax.set_title(r"$n_s=10^{14}$ cm$^{-2}$", fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.0, loc="upper left", bbox_to_anchor=(0.0, 1.0)); ax.set_ylim(0, 1.3); clean(ax); label(ax, "c", -0.22)
    fig.savefig(os.path.join(outdir, "fig3_collective.pdf")); plt.close(fig)


def fig4b():
    Lt = J("lifetimes.json"); C = J("chi2.json")
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw=dict(wspace=0.38))
    ax = axs[0]
    for NA, (col, ls, lab) in NAC.items():
        R = sorted([r for r in Lt["rows"] if r["NA"] == NA], key=lambda r: r["ns"]); ns = np.array([r["ns"] for r in R]) / 1e13
        ax.plot(ns, [r["tau_edge_unscreened_fs"] for r in R], color=col, ls=":", lw=0.9)
        ax.plot(ns, [r["tau_fs"]["unblocked_unscreened"] for r in R], color=col, ls="-.", lw=0.7)
        ax.plot(ns, [r["tau_fs"]["pauli_unscreened"] for r in R], color=col, ls=ls, lw=0.9, marker="o", ms=2.2, mfc="white")
        ax.plot(ns, [r["tau_fs"]["pauli_screened"] for r in R], color=col, ls=ls, marker="o", ms=2.2, label=lab)
    ax.set_xlabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)"); ax.set_ylabel(r"$\tau_{21}$ (fs), LO phonons")
    ax.text(0.97, 0.04, "filled: Pauli blocking + RPA screening\nopen: Pauli blocking, unscreened\ndash-dot: k-averaged, unblocked\ndotted: subband edge, unblocked", transform=ax.transAxes, ha="right", fontsize=5.6, color=INK2)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left"); clean(ax); label(ax, "a", -0.22)
    ax = axs[1]
    for NA, (col, ls, lab) in NAC.items():
        R = sorted([o for o in C["rows"] if o["NA"] == NA], key=lambda o: o["ns"]); ns = np.array([o["ns"] for o in R]) / 1e13
        ax.semilogy(ns, [o["rpa_10"]["peak_nm2V"] for o in R], color=col, ls=ls, marker="o", ms=2.4, label=lab + r", $\gamma$=10 meV")
        ax.semilogy(ns, [o["rpa_30"]["peak_nm2V"] for o in R], color=col, ls=ls, marker="o", ms=2.4, mfc="white", lw=0.8, label=lab + r", $\gamma$=30 meV")
    for g, lsr in (("g10", "--"), ("g30", ":")):
        ax.axhline(C["reference"][g]["peak_sheet_nm2V"], color=INK2, lw=0.7, ls=lsr)
    ax.text(1.3, C["reference"]["g10"]["peak_sheet_nm2V"] * 1.12, "one InGaAs/InAlAs double-well period, 10 meV", fontsize=5.4, color=INK2)
    ax.text(8.3, C["reference"]["g30"]["peak_sheet_nm2V"] * 1.12, "30 meV", fontsize=5.4, color=INK2)
    ax.set_xlabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)"); ax.set_ylabel(r"peak $|\chi^{(2)}_{s}|$ (nm$^2$/V), collective")
    ax.set_ylim(6, 8e3)
    ax.legend(frameon=False, fontsize=5.0, loc="lower center", ncol=2, columnspacing=0.8); clean(ax); label(ax, "b", -0.25)
    fig.savefig(os.path.join(outdir, "fig4_lifetime_chi2.pdf")); plt.close(fig)


def fig5():
    U = J("uq.json")
    UM = J("uq_mean.json")
    for k in U["stats"]:
        U["stats"][k]["factor"] = UM["stats"][k]["factor"]
    qs = [("E21", r"$E_{21}$"), ("median", "collective median"), ("factor", "blue-shift factor"),
          ("chi_s", r"$|\chi^{(2)}_s|_{\max}$ (30 meV)"), ("tau21_fs", r"$\tau_{21}$"), ("margin_cap", "cap margin")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.55, width_ratios=[1.0, 1.25]))
    ax = axs[0]
    for j, (ns, col) in enumerate(((3e13, BLUE), (1e14, PURPLE))):
        rows_ = [r for r in U["rows"] if r["ns"] == ns and r.get("ok")]
        for i, (q, lab) in enumerate(qs[:2]):
            v = np.array([r[q] for r in rows_])
            pos = i * 3 + j
            parts = ax.violinplot([v], positions=[pos], widths=0.8, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.45)
            ax.plot([pos], [np.median(v)], marker="_", color=INK, ms=10)
    ax.set_xticks([0.5, 3.5]); ax.set_xticklabels([r"$E_{21}$", "collective\nmedian"]); ax.set_ylabel("energy (eV)")
    ax.text(0.02, 0.95, r"$3\times10^{13}$", color=BLUE, transform=ax.transAxes, fontsize=6.5)
    ax.text(0.02, 0.87, r"$10^{14}$ cm$^{-2}$", color=PURPLE, transform=ax.transAxes, fontsize=6.5)
    clean(ax); label(ax, "a", -0.22)
    ax = axs[1]
    keys = list(U["ranges"].keys())
    names = {"m": r"$m^*$", "beta": r"$\beta$", "eps_inf": r"$\varepsilon_\infty$", "cbo": "CBO",
             "m_cap": r"$m_{\rm LIO}$", "NA": r"$N_A$", "E_pin": r"$E_{\rm pin}$"}
    M = np.array([[U["stats"]["1e+14"][q]["spearman"][k] for k in keys] for q, _ in qs])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([names[k] for k in keys], fontsize=7)
    ax.set_yticks(range(len(qs))); ax.set_yticklabels([l for _, l in qs], fontsize=6.6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=5.4, color="white" if abs(M[i, j]) > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("Spearman rank correlation", fontsize=6.5)
    ax.set_title(r"$n_s=10^{14}$ cm$^{-2}$", fontsize=7.5)
    label(ax, "b", -0.42)
    fig.savefig(os.path.join(outdir, "fig5_uq.pdf")); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4b, "5": fig5}[w]()
    print("ok")
