module Nelder_mead

function nelder_mead(f, x_start, step = 0.1, no_improve_thr=10e-8,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5)

    simplex = initialize(f, x_start, step)
    prev_best_score = f(x_start)
    no_improv = 0

    iters = 0

    while true
        best_score = simplex[1,4]

        # break after max_iter
        if iters >= max_iter && max_iter > 0
            return simplex[1,:]
        end
        iters += 1

        # break after no_improv_break iterations with no improvement
        if best_score < prev_best_score - no_improve_thr
            no_improv = 0
            prev_best_score = best_score
        else
            no_improv += 1
        end

        if no_improv >= no_improv_break
            return simplex[1,:]
        end

        simplex = run_step(f, simplex, alpha, gamma, rho, sigma)
    end

end

function initialize(f, x_start, step)

    dim = length(x_start)
    score = f(x_start)
    simplex = zeros(dim + 1,dim + 1)
    simplex[1,1:3] = x_start; simplex[1,4] = score

    for i in 1:dim
        x = copy(x_start)
        x[i] += step
        score = f(x)
        simplex[i + 1, 1:3] = x; simplex[i + 1, 4] = score
    end
    simplex = sortslices(simplex, dims=1, lt=(x,y)->isless(x[4],y[4]))
    return simplex

end


function run_step(f, simplex_old, alpha = 1.0, gamma = 2.0, rho = -0.5, sigma = 0.5)

    #sorting
    simplex = copy(simplex_old)
    simplex = sortslices(simplex, dims=1, lt=(x,y)->isless(x[4],y[4]))

    # centroid
    dim = size(simplex)[1] - 1
    x0 = zeros(dim)
    for i in 1:dim
        row = simplex[i,1:3]
        for (i, xi) in enumerate(row)
            x0[i] += xi / dim
        end
    end

    # reflection
    xr = x0 + alpha*(x0 - simplex[end,1:3])
    rscore = f(xr)
    if simplex[1,4] <= rscore < simplex[end-1,4]
        simplex[end,1:3] = xr; simplex[end,4] = rscore
        return simplex
    end

    # expansion
    if rscore < simplex[1,4]
        xe = x0 + gamma*(x0 - simplex[end,1:3])
        escore = f(xe)
        if escore < rscore
            simplex[end,1:3] = xe; simplex[end,4] = escore
            return simplex
        else
            simplex[end,1:3] = xr; simplex[end,4] = rscore
            return simplex
        end
    end

    # contraction
    xc = x0 + rho*(x0 - simplex[end,1:3])
    cscore = f(xc)
    if cscore < simplex[end,4]
        simplex[end,1:3] = xc; simplex[end,4] = cscore
        return simplex
    end

    # reduction
    x1 = simplex[1,1:3]
    new_simplex = zeros(dim + 1, dim + 1)
    for i in 1:dim+1
        redx = x1 + sigma*(simplex[i,1:3] - x1)
        score = f(redx)
        new_simplex[i,1:3] = redx; new_simplex[i,4] = score
    end

    return new_simplex
end

end

# function f(x)
#     return sin(x[1]) * cos(x[2]) * (1. / (abs(x[3]) + 1.6))
# end
#
# function run_multiple()
#     for i in 1:10000
#     nelder_mead(f, [0., 0., 0.1 + i/100000])
#     end
# end
#
# @time begin
#     run_multiple()
# end
#
# @time begin
#     run_multiple()
# end
