import numpy as np
from numpy import linalg as LA
from sympy import *
import time
import copy
import matplotlib.pyplot as plt
from optimizer import sliding
from plotting import *
from numba import jit, jitclass
import numba

ERROR_BOUND = 1e-10

def e_p(r):
    return -1/(1 + r**2/2)

def f_p(r):
    r_sym = Symbol('r_sym')
    e_p = lambda r_sym: -1/(1 + r_sym**2/2)
    f_p = -diff(e_p(r_sym), r_sym)
    f_p = lambdify(r_sym, f_p)
    return f_p(r)

def df_p(r):
    r_sym = Symbol('r_sym')
    e_p = lambda r_sym: -1/(1 + r_sym**2/2)
    f_p = lambda r_sym: -diff(e_p(r_sym), r_sym)
    df_p = diff(f_p(r_sym), r_sym)
    df_p = lambdify(r_sym, df_p)
    return df_p(r)

def d2f_p(r):
    r_sym = Symbol('r_sym')
    e_p = lambda r_sym: -1/(1 + r_sym**2/2)
    f_p = lambda r_sym: -diff(e_p(r_sym), r_sym)
    df_p = lambda r_sym: diff(f_p(r_sym), r_sym)
    d2f_p = diff(df_p(r_sym), r_sym)
    d2f_p = lambdify(r_sym, d2f_p)
    return d2f_p(r)

class two_defects():

    def __init__(self, type = 'Lorentzian'):

        self.type = type
        self._compute_functions()

    def _compute_functions(self):

         g = Symbol('g'); C = Symbol('C');
         X = Symbol('X');  Y = Symbol('Y'); R = np.array([X, Y])
         Delta_X = Symbol('Delta_X'); Delta_Y = Symbol('Delta_Y'); Delta_R = np.array([Delta_X, Delta_Y])
         r1_x = Symbol('r1_x');  r1_y = Symbol('r1_y');
         r2_x = Symbol('r2_x');  r2_y = Symbol('r2_y');
         r1 = np.array([r1_x, r1_y])
         r2 = np.array([r2_x, r2_y])

         energy = self._defect_potential(r1) + self._defect_potential(r2) + self._elastic_energy(r1 - (R - Delta_R/2), r2 - (R + Delta_R/2), C, g)
         force = -np.array([diff(self._defect_potential(r1), var) for var in r1]) - np.array([diff(self._defect_potential(r2), var) for var in r2])
         grad = np.array([diff(energy, var) for var in [*r1, *r2]])
         hess = np.array(hessian(energy,[*r1, *r2]))

         all_vars = [g, C, *r1, *r2, *R, *Delta_R]
         self.energy_eq = lambdify(all_vars,energy)
         self.force_eq = lambdify(all_vars,force)
         self.grad_eq = lambdify(all_vars,grad)
         self.hess_eq = lambdify(all_vars,hess)

    def _defect_potential(self, r):
        '''
            r       symbolic vector of vortex tip position r = [x, y]
            returns potential energy of the vortex
        '''

        if self.type == 'Lorentzian':
            potential = -1/(1 + np.sum(np.power(r,2))/2)

        return potential

    def _elastic_energy(self, u1, u2, C, g):
        '''
            u1, u2  symbolic vectors of vortex displacement
            returns elastic energy of the effective two-vortex system
        '''

        energy = 1/2*C/(1-g**2)*np.sum(u1**2 + u2**2 - 2*g*u1*u2)

        return energy

    def build_system(self, g, kappa, Delta_R):

        #set parameters
        self.g = g
        self.kappa = kappa
        self.Delta_R = Delta_R
        self.C = 0.25/self.kappa #to be replaced by a more generic formula

        self.energy = lambda r, R: self.energy_eq(self.g, self.C, *r, *R, *self.Delta_R)
        self.force = lambda r, R: np.array(self.force_eq(self.g, self.C, *r, *R, *self.Delta_R))
        self.grad = lambda r, R: np.array(self.grad_eq(self.g, self.C, *r, *R, *self.Delta_R))
        self.hess = lambda r, R: np.array(self.hess_eq(self.g, self.C, *r, *R, *self.Delta_R))

    def place_vortices(self,R, initialize_r = False):
        self.R = np.array(R).astype(np.float64)

        if initialize_r:
            self.r = np.array([*R, *R]).astype(np.float64)

def refine_jumps(system, b, r_jump, r_no_jump, R_jumps, R_no_jumps, e_jump, e_no_jumps, lambdas, max_step):

    e_jumps = [e_jump];
    #evaluate missing information
    try:
        delta_R_no_jump = R_no_jumps[-1] - R_no_jumps[-2]
    except:
        delta_R_no_jump = np.infty
    try:
        delta_R_jump = R_jumps[-2] - R_jumps[-1]
    except:
        delta_R_jump = np.infty

    #calculate additional energy for the branch after jump
    delta_R = min(delta_R_jump, delta_R_no_jump)
    r_jump, e_jump, lambda_min, jump, delta_r, (r_trajectory1, r_sliding1) = sliding(system, initial_r = r_jump, R = [R_jumps[-1] - delta_R, b],
        optimizer = 'Hess', max_step = max_step)
    if not jump:
        R_jumps.append(R_jumps[-1] - delta_R);
        e_jumps.append(e_jump)

    if delta_R_no_jump > delta_R_jump:
        #further refinement of the pinned branch
        r_no_jump, e_no_jump, lambda_min, jump, delta_r, (r_trajectory1, r_sliding1) = sliding(system, initial_r = r_no_jump, R = [R_no_jumps[-1] - delta_R_jump, b],
            optimizer = 'Hess', max_step = max_step)
        if not jump:
            R_no_jumps.insert(-1,R_no_jumps[-1] - delta_R_jump); e_no_jumps.insert(-1,e_no_jump); lambdas.insert(-1, lambda_min)

    def slope(x,y):
        return (y[1] - y[0])/(x[1] - x[0])

    if len(R_no_jumps) > 1:
        no_jumps_slope = slope(R_no_jumps[-2:],e_no_jumps[-2:])
        R_last_no_jump = (lambdas[-2]**2*R_no_jumps[-1] - lambdas[-1]**2*R_no_jumps[-2])/(lambdas[-2]**2 - lambdas[-1]**2)
        e_last_no_jump = e_no_jumps[-1] + no_jumps_slope*(R_last_no_jump - R_no_jumps[-1])
        delta_x1 = R_last_no_jump - R_no_jumps[-1]; delta_x2 = R_last_no_jump - R_no_jumps[-2]

        # plot_lambdas(R_no_jumps, lambdas, 0)
        # plt.show()

        if delta_x1 > 0 and delta_x2 > 0:

            #r_last_no_jump = (delta_x2**0.5*r_no_jumps[-1] - delta_x1**0.5*r_no_jumps[-2])/(delta_x2**0.5 - delta_x1**0.5)

            lambdas.append(0)
            R_no_jumps.append(R_last_no_jump)
            #r_no_jumps.append(r_last_no_jump)
            e_no_jumps.append(e_last_no_jump)

            if len(R_jumps) > 1:

                e_jumps_slope = slope(R_jumps[-2:],e_jumps[-2:])
                e_first_jump = e_jumps[-1] + e_jumps_slope*(R_last_no_jump - R_jumps[-1])

                #r_jumps_slope = slope(R_jumps[-2:],r_jumps[-2:])
                #r_first_jump = r_jumps[-1] + r_jumps_slope*(R_last_no_jump - R_jumps[-1])

                R_jumps.append(R_last_no_jump)
                e_jumps.append(e_first_jump)
                #r_jumps.append(r_first_jump)

    return R_jumps, R_no_jumps, e_jumps, e_no_jumps, lambdas

def binary_search(system, b, R_jump, R_no_jump, e_no_jump, r_no_jump, lambda_no_jump, max_step, plotting = None):

    r_no_jumps = []; e_no_jumps = []; R_jumps = []; R_no_jumps = []; lambdas = [];

    R_no_jumps.append(R_no_jump); R_jumps.append(R_jump);
    e_no_jumps.append(e_no_jump);
    r_no_jumps.append(r_no_jump.copy());
    lambdas.append(lambda_no_jump)

    steps = 0

    while R_jump - R_no_jump > 1e-8:

        steps += 1
        R_try = 0.5*(R_jump + R_no_jump)

        r, energy, lambda_min, jump, delta_r, (r_trajectory1, r_sliding1) = sliding(system, initial_r = r_no_jump, R = [R_try, b],
            optimizer = 'Hess', max_step = max_step, write_log = False)

        if not jump:
            R_no_jumps.append(R_try); R_no_jump = R_try
            e_no_jumps.append(energy); e_no_jump = energy
            r_no_jumps.append(r.copy()); r_no_jump = r.copy()
            lambdas.append(lambda_min)
            system.r = r.copy()

        else:
            R_jumps.append(R_try); R_jump = R_try

    return R_jumps, R_no_jumps, e_no_jumps, r_no_jumps, lambdas, steps


def is_jump(system, interval, N, b = 0, sliding_coeff = 0.01):

    step_R = (interval[-1] - interval[0])/(N-1)


    r_values = []; R_values = []; energy_values = []; Delta_e = []; R_jump_ids = [];

    R = interval[0]
    initialize_r = True

    while R < interval[-1]:

        system.place_vortices([R, b], initialize_r)
        initialize_r = False

        r, energy, lambda_min, jump, _, (r_trajectory, r_sliding) = sliding(system, initial_r = system.r.copy(), R = [R, b],
                            optimizer = 'Hess')

        system.r = r.copy()
        R += step_R

        if jump:
            return True

    return False

def force(system,interval,N, b = 0, sliding_coeff = 0.01, optimizer = 'Hess', draw_landscapes = False, plotting = None, message = True):

    step_R = (interval[-1] - interval[0])/(N-1)
    max_step = step_R*sliding_coeff

    r_values = []; R_values = []; energy_values = []; Delta_e = []; R_jump_ids = [];

    R = interval[0]
    initialize_r = True

    while R < interval[-1]:

        system.place_vortices([R, b], initialize_r)
        initialize_r = False

        r, energy, lambda_min, jump, _, (r_trajectory, r_sliding) = sliding(system, initial_r = system.r.copy(), R = [R, b],
                            max_step = max_step, optimizer = optimizer)


        if not jump:

            R_values.append(R); R_no_jump = R
            energy_values.append(energy); e_no_jump = energy
            r_values.append(r.copy())
            lambda_no_jump = lambda_min
            system.r = r.copy()
            R += step_R

        else:

            R_jumps, R_no_jumps, e_no_jumps, r_no_jumps, lambdas, steps = binary_search(system, b, R, R_no_jump, e_no_jump, system.r.copy(), lambda_no_jump, max_step, plotting)

            R_values.extend(R_no_jumps);
            R_jump_ids.append(len(R_values) - 1);
            energy_values.extend(e_no_jumps);
            r_values.extend(r_no_jumps);

            r_jump, e_jump, _, jump, _, (r_trajectory, r_sliding) = sliding(system, initial_r = r_no_jumps[-1], R = [R_jumps[-1], b],
                                max_step = step_R*sliding_coeff, optimizer = optimizer, run_to_end = True) #only here we actually do the sliding
            if not jump:
                raise Exception('Did not jump after the binary search. Shutting down.')

            R_values.extend([np.nan, R_jumps[-1]])
            energy_values.extend([np.nan, e_jump])
            r_values.extend([np.array(4*[np.nan]),r_jump.copy()])

            R_no_jump = R_jumps[-1]
            R = R_no_jump + step_R
            system.r = r_jump.copy()

            R_jumps, R_no_jumps, e_jumps, e_no_jumps, lambdas = refine_jumps(system, b, r_jump, r_no_jumps[-1], R_jumps, R_no_jumps, e_jump, e_no_jumps, lambdas, max_step)

            Delta_e_new = e_no_jumps[-1] - e_jumps[-1]
            #last check if something went wrong
            if  Delta_e_new < 0:
                Delta_e.append(np.nan)
            else:
                Delta_e.append(Delta_e_new)

            if message:
                print('Jump at R = {}, Delta_e = {}, steps = {}'.format(R_values[R_jump_ids[-1]],Delta_e[-1],steps))

    return np.array(R_values), np.array(energy_values), np.stack(r_values), Delta_e, R_jump_ids
