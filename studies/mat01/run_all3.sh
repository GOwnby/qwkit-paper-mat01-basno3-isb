#!/bin/sh
# MAT-01 UQ (chi2 at gamma = 30 meV), then the boundary-fix reruns of the other papers.
cd "$(dirname "$0")"
python uq.py > uq.log 2>&1
echo MAT01-DONE > chain_done.flag
cd ..
for d in n1_paper n1 nl02 nl05 ms02; do mkdir -p $d/pre_bcfix; cp $d/*.json $d/*.log $d/pre_bcfix/ 2>/dev/null; done
for s in rerun_nl05.sh rerun_n1.sh rerun_nl02.sh rerun_ms02.sh; do
  echo "start $s $(date)" >> queue_bcfix.log
  sh $s >> queue_bcfix.log 2>&1 || echo "FAILED $s" >> queue_bcfix.log
  echo "end $s $(date)" >> queue_bcfix.log
done
echo QUEUE-DONE >> queue_bcfix.log
