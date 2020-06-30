#!/bin/bash

python hdmm.py --dataset=adult --workload=24 --epsilon=0.1
python hdmm.py --dataset=adult --workload=24 --epsilon=0.15
python hdmm.py --dataset=adult --workload=24 --epsilon=0.2
python hdmm.py --dataset=adult --workload=24 --epsilon=0.25
python hdmm.py --dataset=adult --workload=24 --epsilon=0.5
python hdmm.py --dataset=adult --workload=24 --epsilon=1
python hdmm.py --dataset=adult --workload=24 --epsilon=20000
