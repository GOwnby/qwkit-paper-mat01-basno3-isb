#!/bin/sh
# MAT-01 full rerun after the interface Poisson fix (12 Oct 2026, evening), two lanes, then the other papers.
cd "$(dirname "$0")"
( python linear.py > linear.log 2>&1; python chi2.py > chi2.log 2>&1; python checks.py > checks.log 2>&1
  python lifetimes.py > lifetimes.log 2>&1; python uq.py > uq.log 2>&1; touch laneA.flag ) &
( python -W ignore gating.py > gating.log 2>&1; python -W ignore extras.py > extras.log 2>&1
  python -W ignore capmodes.py > capmodes.log 2>&1; python -W ignore conv2.py > conv2.log 2>&1
  python -W ignore fsum.py > fsum.log 2>&1; python -W ignore kane_check.py > kane_check.log 2>&1
  python -W ignore kane_rpa.py > kane_rpa.log 2>&1; python -W ignore basis_clean.py > basis_clean.log 2>&1; touch laneB.flag ) &
wait
echo MAT01-DONE > chain_done.flag
cd ..
for d in n1_paper n1 nl02 nl05 ms02; do mkdir -p $d/pre_bcfix; cp $d/*.json $d/*.log $d/pre_bcfix/ 2>/dev/null; done
for s in rerun_nl05.sh rerun_n1.sh rerun_nl02.sh rerun_ms02.sh; do
  echo "start $s $(date)" >> queue_bcfix.log
  sh $s >> queue_bcfix.log 2>&1 || echo "FAILED $s" >> queue_bcfix.log
  echo "end $s $(date)" >> queue_bcfix.log
done
echo QUEUE-DONE >> queue_bcfix.log
