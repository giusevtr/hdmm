#!/bin/bash
wl=120
ds="loans"
python hdmm.py --dataset=$ds --workload=15 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=30 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=60 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=120 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=180 --epsilon=0.1
#python hdmm.py --dataset=$ds --workload=$wl --epsilon=0.15
#python hdmm.py --dataset=$ds --workload=$wl --epsilon=0.2
#python hdmm.py --dataset=$ds --workload=$wl --epsilon=0.25
#python hdmm.py --dataset=$ds --workload=$wl --epsilon=0.5
#python hdmm.py --dataset=$ds --workload=$wl --epsilon=1
#python hdmm.py --dataset=$ds --workload=$wl --epsilon=20000
