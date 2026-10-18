"""MAT-01 check: f-sum rule of the k-resolved response. The integrated absorbed power (loss) over 0.02-4 eV is the
same for the single-particle and collective spectra when the sum rule holds (the Hartree coupling redistributes but
does not create oscillator strength). Two cases: N_A = 5e19, 1e14 cm^-2 and N_A = 1e18, 1e13 cm^-2, 300 K."""
from common import *

HW = np.arange(0.02, 4.0001, 0.0025)
for NA, ns in ((5e19, 1e14), (1e18, 1e13), (5e19, 3e13)):
    p = dict(BASE); p["NA"] = NA
    r = solve(ns, p)
    tot = {}
    for kern in (False, True):
        k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=200, gamma=0.010,
                                    eps_b=p["eps_inf"], kernel=kern, zc=0.0, occ_tol=1e-5)
        L = np.array([k.loss(h) for h in HW])
        tot[kern] = (float(np.sum(L)), float(np.sum(L[HW >= 0.1])))
    print(f"NA {NA:.0e} ns {ns:.0e}: integrated loss RPA/bare over 0.02-4 eV {tot[True][0]/tot[False][0]:.4f}, "
          f"over 0.1-4 eV {tot[True][1]/tot[False][1]:.4f} (constant eps_inf background)", flush=True)
