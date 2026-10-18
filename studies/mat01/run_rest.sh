#!/bin/sh
cd "$(dirname "$0")"
python lifetimes.py > lifetimes.log 2>&1
python chi2.py > chi2.log 2>&1
python checks.py > checks.log 2>&1
python uq.py > uq.log 2>&1
