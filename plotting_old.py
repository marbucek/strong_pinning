
def draw_landscape(system, R, r_start_collection, r_end_collection, R_branch, e_branch, r_trajectories, r_slides, before = 0, extension = 0.2):
    """r_start_collection  multiple starting points
       r_end_collection    multiple ending points
    """

    #print(f'Higher point: r = {r[0]}, delta_r = {delta_r[0]}')
    #print(f'Lower point: r = {r2[0]}, delta_r = {delta_r[0]}')
    #print(f'Difference in r = {r2[0] - r[0]}')
    #draw the local energy landscape

    #try:
    #    print(f'In draw_landscape: r_start = {r_start_collection[1]}, r_end = {r_end_collection[1]}')
    #except:
    #    pass

    r_plot_collection = [[(1-c)*r_start + c*r_end for c in np.linspace(-before,1 + extension,100)]
                for r_start, r_end in zip(r_start_collection, r_end_collection)]
    energy_plot_collection = [[system.energy(r,R) for r in r_plot] for r_plot, shift in zip(r_plot_collection, [0,0.01])]

    #plot sliding
    plt.figure(figsize=(15,3))
    for id, subplot_id in zip([0,2],[141,142]):
        for r_start, r_end, r_plot, energy_plot, shift in zip(r_start_collection, r_end_collection, r_plot_collection, energy_plot_collection, [0,0.01]):
            energy_start = system.energy(r_start,R)
            energy_end = system.energy(r_end,R)
            plt.subplot(subplot_id)
            plt.scatter([r_start[id], r_end[id]], [energy_start, energy_end],c= ['r','g'])
            plt.plot([r[id] for r in r_plot], energy_plot)

        x_limits = [
                    min([min([r[id] for r in r_plot]) for r_plot in r_plot_collection]),
                    max([max([r[id] for r in r_plot]) for r_plot in r_plot_collection])
                    ]
        x_span = x_limits[-1] - x_limits[0]
        x_limits[0] += -0.1*x_span
        x_limits[-1] += 0.1*x_span

        y_limits = [
                    min([min(energy_plot) for energy_plot in energy_plot_collection]),
                    max([max(energy_plot) for energy_plot in energy_plot_collection])
                    ]
        y_span = y_limits[-1] - y_limits[0]
        y_limits[0] += -0.1*y_span
        y_limits[-1] += 0.1*y_span

        plt.axis([*x_limits, *y_limits])

    #plot branches
    plt.subplot(143)
    for R, e in zip(R_branch, e_branch):

        plt.plot(R, e)

        try:
            slope = (e[-1] - e[-2])/(R[-1] - R[-2])
            plt.plot([R[-1], R[-1] + 0.1], [e[-1], e[-1] + slope*0.1],c='r')
            plt.scatter([R[-1]],[e[-1]])
        except:
            pass

    if len(R_branch) == 1:
        x_limits = [min([min(R) for R in R_branch]), max([max(R) for R in R_branch])]
        x_span = x_limits[-1] - x_limits[0]
        x_limits[0] += -0.1*x_span
        x_limits[-1] += 0.1*x_span

        y_limits = [min([min(e) for e in e_branch]), max([max(e) for e in e_branch])]
        y_span = y_limits[-1] - y_limits[0]
        y_limits[0] += -0.1*y_span
        y_limits[-1] += 0.1*y_span
    else:
        #x_limits = [min([max(R) for R in R_branch]), max([min(R) for R in R_branch])]
        x_limits = [min([min(R[-2:]) for R in R_branch]), max([max(R[-2:]) for R in R_branch])]
        x_span = x_limits[-1] - x_limits[0]
        x_limits[0] += -0.5*x_span
        x_limits[-1] += 0.5*x_span

        #y_limits = [min([max(e) for e in e_branch]), max([min(e) for e in e_branch])]
        y_limits = [min([min(e[-2:]) for e in e_branch]), max([max(e[-2:]) for e in e_branch])]
        y_span = y_limits[-1] - y_limits[0]
        y_limits[0] += -0.1*y_span
        y_limits[-1] += 0.1*y_span

    plt.axis([*x_limits, *y_limits])



    plt.subplot(144)
    for r_trajectory in r_trajectories:
        plt.plot([r[0] for r in r_trajectory],[r[2] for r in r_trajectory])
        plt.scatter([r[0] for r in r_trajectory[0::1000]],[r[2] for r in r_trajectory[0::1000]])
        plt.scatter([r_trajectory[id][0] for id in [0,-1]],[r_trajectory[id][2] for id in [0,-1]],c=['r','g'])

    for r_slide in r_slides:
        plt.plot([r[0] for r in r_slide],[r[2] for r in r_slide],c = 'r')

    plt.show()

    #plot branches
