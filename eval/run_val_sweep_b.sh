#!/bin/sh
# Validation sweep, phase B (after phase A showed the global stratified context beats per-species contexts).
set -u
cd "$(dirname "$0")/.."
run() { name=$1; shift; echo "== $name $*"; ./run_tg.sh "$name" eval/backtest.py --split val "$@"; echo "exit=$? $(systemctl show tg-$name -p Result 2>/dev/null)"; }
run v-persp-sm   --strategy per_species --features full --n-ctx 1000 --n-est 4 --name tabpfn_full_per_species_smoothed_c1000_e4_s1
run v-strat-core --strategy stratified  --features core --n-ctx 1000 --n-est 4
run v-strat-noan --strategy stratified  --features full_no_anom --n-ctx 1000 --n-est 4
run v-strat-s3   --strategy stratified  --features full --n-ctx 1000 --n-est 4 --subsamples 3
run v-strat-e8   --strategy stratified  --features full --n-ctx 1000 --n-est 8
run v-local-g80  --strategy local       --features full --n-ctx 1000 --n-est 4 --group-sample 80
run v-strat-g80  --strategy stratified  --features full --n-ctx 1000 --n-est 4 --group-sample 80
