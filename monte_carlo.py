import numpy as np

def integrate(fun, bounds, measure = None, sigma = 0.5, N = 1000):

    def measure_unif(x):
        volume = 1
        for bound in bounds:
            volume *= bound[-1] - bound[0]
        return 1/volume

    if measure == None:
        measure = measure_unif

    #print(measure)

    while 1:
        x = np.random.uniform(size = 2)
        f = fun(x)
        g = measure(x)
        if f > 0:
            break

    x_out = [x.copy()]; f_out = [f]; g_out = [g]
    for i in range(N - 1):
        delta_x = np.random.normal(0, sigma, len(bounds))
        #flow over the edges
        x_new = []
        for xi, delta_xi, bound in zip(x, delta_x, bounds):
            x_new.append((xi + delta_xi - bound[0]) % (bound[-1] - bound[0]) + bound[0])
        x_new = np.array(x_new)
        p_accept = np.min([measure(x_new)/measure(x), 1])

        #accept new point with probability p_accept, otherwise re-use the current point
        if np.random.choice([True, False], p = [p_accept, 1 - p_accept]):
            x = x_new.copy()

        f = fun(x); g = measure(x)
        x_out.append(x.copy()); f_out.append(f); g_out.append(g)

    if measure is measure_unif:
        return np.mean(np.array(f_out)/np.array(g_out))
    else:
        return np.mean(np.array(f_out)/np.array(g_out))*integrate(measure, bounds, N = N)
