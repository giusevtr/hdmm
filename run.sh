#!/bin/bash
ds="loans"
python hdmm.py --dataset=$ds --workload=32   --marginal=3 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=128  --marginal=3 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=256  --marginal=3 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=512  --marginal=3 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=1024 --marginal=3 --epsilon=0.1

python hdmm.py --dataset=$ds --workload=32   --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=128  --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=256  --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=512  --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=1024 --marginal=5 --epsilon=0.1


ds="adult"
python hdmm.py --dataset=$ds --workload=32   --marginal=3 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=64   --marginal=3 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=128  --marginal=3 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=256  --marginal=3 --epsilon=0.1

python hdmm.py --dataset=$ds --workload=32   --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=64   --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=128  --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=256  --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=512  --marginal=5 --epsilon=0.1
python hdmm.py --dataset=$ds --workload=1024 --marginal=5 --epsilon=0.1

