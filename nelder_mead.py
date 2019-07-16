import copy

'''
    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    simplex = initialize(f, x_start, step)
    prev_best = f(x_start)
    no_improv = 0

    iters = 0

    while 1:
        # order
        best = simplex[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return simplex[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print(f'{best} best so far:')

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return simplex[0]

        dim = len(x_start)
        simplex = run_step(f, simplex, alpha, gamma, rho, sigma)
        #print(simplex)

def initialize(f, x_start, step):

    dim = len(x_start)
    score = f(x_start)
    simplex = [[x_start, score]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        simplex.append([x, score])

    simplex.sort(key=lambda x: x[1])

    return simplex

def run_step(f, simplex_old, alpha=1., gamma=2., rho=-0.5, sigma=0.5):

    #sorting
    simplex = simplex_old.copy()
    simplex.sort(key=lambda x: x[1])

    # centroid
    dim = len(simplex[0][0])
    x0 = [0.] * dim
    for tup in simplex[:-1]:
        for i, c in enumerate(tup[0]):
            x0[i] += c / (len(simplex)-1)

    # reflection
    xr = x0 + alpha*(x0 - simplex[-1][0])
    rscore = f(xr)
    if simplex[0][1] <= rscore < simplex[-2][1]:
        del simplex[-1]
        simplex.append([xr, rscore])
        return simplex

    # expansion
    if rscore < simplex[0][1]:
        xe = x0 + gamma*(x0 - simplex[-1][0])
        escore = f(xe)
        if escore < rscore:
            del simplex[-1]
            simplex.append([xe, escore])
            return simplex
        else:
            del simplex[-1]
            simplex.append([xr, rscore])
            return simplex

    # contraction
    xc = x0 + rho*(x0 - simplex[-1][0])
    cscore = f(xc)
    if cscore < simplex[-1][1]:
        del simplex[-1]
        simplex.append([xc, cscore])
        return simplex

    # reduction
    x1 = simplex[0][0]
    new_simplex = []
    for tup in simplex:
        redx = x1 + sigma*(tup[0] - x1)
        score = f(redx)
        new_simplex.append([redx, score])
    simplex = new_simplex

    #sorting

    return simplex


if __name__ == "__main__":
    # test
    import math
    import numpy as np

    def f(x):
        return math.sin(x[0]) * math.cos(x[1]) * (1. / (abs(x[2]) + 1))

    print(nelder_mead(f, np.array([0., 0., 0.])))
