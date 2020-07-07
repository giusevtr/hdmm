#!/bin/bash
ds="loans"
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.15
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.2
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.25
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.5
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=1.0

python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.15
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.2
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.25
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.5
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=1.0


ds="adult"
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.15
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.2
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.25
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.5
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=1.0

python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.15
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.2
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.25
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.5
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=1.0

