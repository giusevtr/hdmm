from mbi import mechanism, FactoredInference
import benchmarks
from IPython import embed
import numpy as np
from scipy.sparse.linalg import lsmr
import argparse
from scipy.stats import norm, laplace

def run(dataset,measurements, workloads,  eps=1.0, delta=0.0, bounded=True, engine='MD',
        options={}, iters=10000, seed=None, metric='L2', elim_order=None, frequency=1):
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

        print("=====")
        print("x.shape =", x.shape)
        print("a.shape =", a.shape)
        # print(y)
        # A_inv = np.linalg.pinv(A)
        # print("A_inv.shape =", A_inv.shape)
        # x_bar = lsmr(A, y)[0]

        local_ls[proj] = lsmr(A, y)[0]

    answers = []
    for proj, W in workloads:
        ans = W.dot(local_ls[proj])
        answers.append((ans, proj, W))

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
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--save', action='store_true', help='save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data, measurements, workloads = benchmarks.random_hdmm(args.dataset, args.workload)
    N = data.df.shape[0]
    # model, log, answers = mechanism.run(data, measurements, eps=args.epsilon, delta=1.0/N**2, frequency=50, seed=args.seed, iters=args.iters)
    answers = run(data,  measurements, workloads, eps=args.epsilon, delta=1.0/N**2, frequency=50, seed=args.seed, iters=args.iters)


    print("data.domain", data.domain)
    error_1 = []
    error_2 = []
    for (ans, proj, W) in answers:
        print("proj", proj)
        # W.dot(data.datavector())
        # proj = (A,B,D)
        # true[0] = (A=0 B=0 C=0)
        true = W.dot(data.project(proj).datavector())
        fake = ans
        # fake = ans / N
        print("true: ", true[:10])
        print("true.dim = ", true.shape)
        print("true.sum: ", true.sum())
        # print("ans: ", ans[:10]/N)
        print("fake: ", fake[:10])
        print("fake.dim: ", fake.shape)
        print("fake.sum: ", fake.sum())
        error_l_inf = np.max(np.abs(fake - true))
        # error_l_inf = np.max(np.abs(ans - true)) / N
        err = np.abs(fake - true).sum()
        error_1.append(error_l_inf)
        error_2.append(err)
        # print("true.sum = ", (true).sum())
        # print("abs(true).sum = ", np.abs(true).sum())

    max_error = np.max(error_1)
    mean_error = np.mean(error_2)

    print("eps = {}\tmax_error={:.4f}".format(args.epsilon, max_error))
    # print("mean_error", mean_error)
    # path = 'results/hdmm.csv'
    # with open(path, 'a') as f:
    #     f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s \n' % (args.dataset,args.seed,args.epsilon,err_hdmm1, err_pgm1, err_hdmm2, err_pgm2, err_pgm1a, err_pgm2a))
    #
