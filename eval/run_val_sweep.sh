#!/bin/sh
# Validation sweep (train 2018-2023, evaluate 2024) on the CPU box, one TabPFN job at a time,
# each under a hard 800 MB memory cap (run_tg.sh wraps systemd-run). Usage: sh eval/run_val_sweep.sh <phase>
set -u
cd "$(dirname "$0")/.."
R=./run_tg.sh
run() { name=$1; shift; echo "== $name $*"; $R "$name" eval/backtest.py --split val "$@"; echo "exit=$? $(systemctl show tg-$name -p Result 2>/dev/null)"; }
case "${1:-a}" in
  a)
    run v-strat    --strategy stratified  --features full     --n-ctx 1000 --n-est 4
    run v-persp    --strategy per_species --features full     --n-ctx 1000 --n-est 4
    run v-strat-cal --strategy stratified --features calendar --n-ctx 1000 --n-est 4
    run v-persp-cal --strategy per_species --features calendar --n-ctx 1000 --n-est 4
    ;;
  *)
    shift
    run "$@"
    ;;
esac
