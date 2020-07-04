import benchmarks
import os
import numpy as np
from scipy.sparse.linalg import lsmr
import argparse
from scipy.stats import norm, laplace
import time
import pandas as pd


def run(dataset,measurements, workloads,  eps=1.0, delta=0.0, bounded=True, seed=None):
    """
    Run a mechanism that measures the given measurements and runs inference.
    This is a convenience method for running end-to-end experiments.
    """
    state = np.random.RandomState(seed)
    l1 = 0
    l2 = 0
    for _, Q in measurements:
        l1 += np.abs(Q).sum(axis=0).max()
        try:
            l2 += Q.power(2).sum(axis=0).max()  # for spares matrices
        except:
            l2 += np.square(Q).sum(axis=0).max()  # for dense matrices

    if bounded:
        total = dataset.df.shape[0]
        l1 *= 2
        l2 *= 2

    if delta > 0:
        noise = norm(loc=0, scale=np.sqrt(l2 * 2 * np.log(2 / delta)) / eps)
    else:
        noise = laplace(loc=0, scale=l1 / eps)

    x_bar_answers = []
    local_ls = {}
    for proj, A in measurements:
        x = dataset.project(proj).datavector()
        z = noise.rvs(size=A.shape[0], random_state=state)
        a = A.dot(x)
        y = a + z
        local_ls[proj] = lsmr(A, y)[0]

    answers = []
    for proj, W in workloads:
        answers.append((local_ls[proj], proj, W))

    return answers


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = 'adult'
    params['workload'] = 15
    params['iters'] = 10000
    params['epsilon'] = 1.0
    params['seed'] = 0
    params['save'] = None

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', choices=['adult','titanic','msnbc','loans','nltcs','fire','stroke','salary'], help='dataset to use')
    parser.add_argument('--workload', type=int, help='number of marginals in workload')
    parser.add_argument('--marginal', type=int, help='number of marginals in workload')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--save', action='store_true', help='save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    print(args)
    start_time = time.time()
    data, measurements, workloads = benchmarks.random_hdmm(args.dataset, args.workload, args.marginal)
    N = data.df.shape[0]
    # model, log, answers = mechanism.run(data, measurements, eps=args.epsilon, delta=1.0/N**2, frequency=50, seed=args.seed, iters=args.iters)
    answers = run(data,  measurements, workloads, eps=args.epsilon, delta=1.0/N**2, seed=args.seed)

    error_1 = []
    error_2 = []
    for (y, proj, W) in answers:
        # Error type 1
        data_proj = data.project(proj).datavector()
        y_normalized = y.copy()/N
        error_type_1 = np.max(np.abs(data_proj/N - y_normalized))
        error_1.append(error_type_1)

        # Error type 2
        true = W.dot(data_proj)
        est = W.dot(y)
        error_type_2 = np.mean(np.abs(true - est))
        error_2.append(error_type_2)

    max_error_1 = np.max(error_1)
    max_error_2 = np.mean(error_2)

    print("eps = {}\terror_1={:.4f}\terror_2={:.4f},\time={:.4f}".format(args.epsilon, max_error_1, max_error_2, time.time()-start_time))
    file_name = "Results/{}_{}_{}.csv".format(args.dataset, args.workload, args.marginal)
    print("Saving ", file_name)
    names = ["epsilon", "error", "runtime"]
    final_df = pd.DataFrame([args.epsilon, max_error_1, time.time()-start_time], columns=names)
    if os.path.exists(file_name):
        dfprev = pd.read_csv(file_name)
        final_df = final_df.append(dfprev, sort=False)
    final_df.to_csv(file_name)

    # print("mean_error", mean_error)
    # path = 'results/hdmm.csv'
    # with open(path, 'a') as f:
    #     f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s \n' % (args.dataset,args.seed,args.epsilon,err_hdmm1, err_pgm1, err_hdmm2, err_pgm2, err_pgm1a, err_pgm2a))
    #
