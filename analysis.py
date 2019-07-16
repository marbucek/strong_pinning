from defect import two_defects, is_jump
import numpy as np
import time

def Delta_max(g, kappa, theta, low = 0, high = 1, R_bound = 10, N = 200):

    system = two_defects()

    error = 1e-5;
    while high - low > error:
        Delta = (high + low)/2
        system.build_system(g, kappa, [Delta*np.cos(theta), Delta*np.sin(theta)])
        jump = is_jump(system, [-R_bound, R_bound], N, b = 0, sliding_coeff = 0.01)
        if jump:
            low = Delta
        else:
            high = Delta

    return (high + low)/2

def plot_Delta_max(g, kappa, N = 100):

    theta_values = np.linspace(0, np.pi/2, N)
    Delta = []
    for theta in theta_values:
        Delta.append(Delta_max(g, kappa, theta))

    plt.plot()

def test_scaling(epsilon, kappa, theta, N = 10):

    g0 = 1/kappa - 1
    g_values = np.linspace(g0, g0 + epsilon, N + 1)[1:]
    Deltas = []
    for g in g_values:
        print(f'Testing scaling for g = {g}')
        Deltas.append(Delta_max(g, kappa, theta))

    return g_values - g0, Deltas
