#!/bin/bash
# Regenerate every MAT-01 number and figure (paper_mat01/). Two lanes on 2 cores, about 4 h (most of it uq.py).
set -e
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd)
cd "$(dirname "$0")/mat01"
( python3 conv.py > conv.log 2>&1; python3 linear.py > linear.log 2>&1; python3 chi2.py > chi2.log 2>&1
  python3 checks.py > checks.log 2>&1; python3 lifetimes.py > lifetimes.log 2>&1
  python3 uq.py > uq.log 2>&1; python3 uq_mean.py > uq_mean.log 2>&1 ) &                 # lane A
( python3 -W ignore gating.py > gating.log 2>&1; python3 -W ignore extras.py > extras.log 2>&1
  python3 -W ignore meanmode.py > meanmode.log 2>&1; python3 -W ignore radiative.py > radiative.log 2>&1
  python3 -W ignore capmodes.py > capmodes.log 2>&1; python3 -W ignore capmodes2.py > capmodes2.log 2>&1
  python3 -W ignore conv2.py > conv2.log 2>&1; python3 -W ignore fsum.py > fsum.log 2>&1
  python3 -W ignore kane_check.py > kane_check.log 2>&1; python3 -W ignore kane_rpa.py > kane_rpa.log 2>&1
  python3 -W ignore basis_clean.py > basis_clean.log 2>&1 ) &                               # lane B
wait
python3 collect_numbers.py > collect_numbers.log 2>&1
python3 appendix_table.py > appendix_table.log 2>&1
python3 paper_figures.py
cd ../../paper_mat01 && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
cp main.pdf ../MAT01_paper.pdf
