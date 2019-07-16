import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
import nelder_mead as ND
from plotting import plot_optimizer_trajectories

ERROR_r = 1e-10

def sliding(system, initial_r, R, optimizer, max_step, write_log = False, run_to_end = False):

    delta_r_norm = np.infty
    delta_r = np.nan
    r_error = np.infty
    jump = False

    energy_fun = lambda r_var: system.energy(r_var, R)
    grad_fun = lambda r_var: system.grad(r_var, R)
    hess_fun = lambda r_var: system.hess(r_var, R)

    r_trajectory = []; r_sliding = []

    #r = copy.copy(initial_r)
    r = initial_r.copy()
    #old_r = initial_r
    energy_old = energy_fun(r)
    energy_new = -np.infty
    step_iter = 0
    r_trajectory.append(r.copy())
    r_sliding.append([np.nan, np.nan, np.nan, np.nan])
    #while energy_old - energy_new > ERROR_BOUND:

    ND_steps = 0; H_steps = 0
    while r_error > ERROR_r:

        step_iter += 1
        grad = grad_fun(r)
        hess = hess_fun(r)
        energy_old = energy_fun(r)
        grad_norm = LA.norm(grad)
        eigenvalues, _ = LA.eig(hess)

        if min(eigenvalues) < 0:
            jump = True
            if not run_to_end:
                return None, None, None, jump, None, (None, None)

        if optimizer == 'Hess':
            H_steps += 1
            if min(eigenvalues) > 0:
                delta_r = -np.matmul(LA.inv(hess),grad)
                delta_r_norm = LA.norm(delta_r)
                r_error = delta_r_norm
                if delta_r_norm > max_step:
                    delta_r = delta_r/delta_r_norm*max_step
                    delta_r_norm = max_step
                r += delta_r
                # if write_log and H_steps % 100 == 0:
                #     print(f'Hess minimization, step_size = {delta_r_norm}')
                energy_new = energy_fun(r)
                r_error = delta_r_norm

            if min(eigenvalues) < 0:
                if write_log:
                    print(f'ND minimization, step = {ND_steps}, step_size = {max_step}')
                if ND_steps == 0:
                    simplex = ND.initialize(energy_fun, r, step = max_step)

                simplex = ND.run_step(energy_fun, simplex)
                r = simplex[0][0]

                ND_steps += 1
                r_error = np.infty

                if ND_steps > 1000:
                    break;

        elif optimizer == 'ND':
            pass
            #to be continued

        if min(eigenvalues) > 0:
            r_trajectory.append(r.copy())
            r_sliding.append([np.nan, np.nan, np.nan, np.nan])
        else:
            r_trajectory.append([np.nan, np.nan, np.nan, np.nan])
            r_sliding.append(r.copy())

    return r, energy_new, min(eigenvalues), jump, delta_r, (r_trajectory, r_sliding)
