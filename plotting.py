import numpy as np
from matplotlib import pyplot as plt

def plot_optimizer_trajectories(r_trajectory, r_sliding):

    for id, (r1, r2) in enumerate(zip(r_trajectory, r_sliding)):
        if id >= 0:
            if np.isnan(r1[0]) and not np.isnan(r_trajectory[id - 1][0]):
                r_sliding[id - 1] = r_trajectory[id - 1]
            if np.isnan(r2[0]) and not np.isnan(r_sliding[id - 1][0]):
                r_trajectory[id - 1] = r_sliding[id - 1]

    plt.plot([r[0] for r in r_trajectory],[r[2] for r in r_trajectory], c = 'b')
    plt.plot([r[0] for r in r_sliding],[r[2] for r in r_sliding],c = 'r')

    if not np.isnan(r_trajectory[0][0]):
        r_start = r_trajectory[0]
    else:
        r_start = r_sliding[0]
    if not np.isnan(r_trajectory[-1][0]):
        r_end = r_trajectory[-1]
    else:
        r_end = r_sliding[-1]

    plt.scatter(r_start[0],r_start[2],c = 'b')
    plt.scatter(r_end[0],r_end[2], c = 'g')

    plt.xlabel('$r_1$')
    plt.ylabel('$r_2$')


def plot_branches(R_jumps, R_no_jumps, e_jumps, e_no_jumps, N):

    plt.plot(R_no_jumps[N:], e_no_jumps[N:])
    plt.scatter(R_no_jumps[-1], e_no_jumps[-1])
    plt.plot(R_jumps[N:], e_jumps[N:])
    plt.scatter(R_jumps[-1], e_jumps[-1])
    plt.xlabel('$R$')
    plt.ylabel('$e_\mathrm{pin}$')


def plot_lambdas(R_no_jumps, lambdas, N):

    plt.plot(R_no_jumps[N:], lambdas[N:])
    plt.scatter(R_no_jumps[N:], lambdas[N:])
    plt.xlabel('$R$')
    plt.ylabel('$\lambda$')


def plot_bsearch_step(r_trajectory, r_sliding, R_jumps, R_no_jumps, e_jumps, e_no_jumps, lambdas, jump, N):

    plt.figure(figsize=(15,3))

    plt.subplot(131)
    plot_optimizer_trajectories(r_trajectory, r_sliding)

    plt.subplot(132)
    plot_branches(R_jumps, R_no_jumps, e_jumps, e_no_jumps, N)

    plt.subplot(133)
    plot_lambdas(R_no_jumps, lambdas, N)

    plt.show()

    print(f'Jump = {jump}, Delta_e = {e_no_jumps[-1] - e_jumps[-1]}')
