include("./functions.jl")
import .Functions
using LinearAlgebra

function ND_initialize(f, x_start, step)

    dim = length(x_start)
    score = f(x_start)
    simplex = zeros(dim + 1,dim + 1)
    simplex = [[x_start, score]]

    for i in 1:dim
        x = copy(x_start)
        x[i] += step
        score = f(x)
        append!(simplex, [[x, score]])
    end

    sort!(simplex, by = x -> x[2])
    return simplex

end

function ND_run_step(f, simplex_old, alpha = 1.0, gamma = 2.0, rho = -0.5, sigma = 0.5)

    #sorting
    simplex = copy(simplex_old)
    sort!(simplex, by = x -> x[2])

    # centroid
    dim = length(simplex[1][1])
    x0 = zeros(dim)
    for tup in simplex[1:end-1]
        for (i, xi) in enumerate(tup[1])
            x0[i] += xi / (length(simplex)-1)
        end
    end

    # reflection
    xr = x0 + alpha*(x0 - simplex[end][1])
    rscore = f(xr)
    if simplex[1][2] <= rscore < simplex[end-1][2]
        pop!(simplex)
        append!(simplex, [[xr, rscore]])
        return simplex
    end

    # expansion
    if rscore < simplex[1][2]
        xe = x0 + gamma*(x0 - simplex[end][1])
        escore = f(xe)
        if escore < rscore
            pop!(simplex)
            append!(simplex, [[xe, escore]])
            return simplex
        else
            pop!(simplex)
            append!(simplex, [[xr, rscore]])
            return simplex
        end
    end

    # contraction
    xc = x0 + rho*(x0 - simplex[end][1])
    cscore = f(xc)
    if cscore < simplex[end][2]
        pop!(simplex)
        append!(simplex, [[xc, cscore]])
        return simplex
    end

    # reduction
    x1 = simplex[1][1]
    new_simplex = []
    for tup in simplex
        redx = x1 + sigma*(tup[1] - x1)
        score = f(redx)
        new_simplex.append([[redx, score]])
    end
    simplex = new_simplex

    return simplex
end

function sliding(energy_f, grad, hess, initial_r, R, max_step;
    min_error = 1e-6, no_improv_thr = 1e-6, max_no_improve = 1000, max_nd_steps = 1000,
    run_to_end = false)

    energy(r) = energy_f(r,R)

    r = copy(initial_r)
    e = energy(r)

    ND_steps = 0
    jump = false
    while true

        Hess = hess(r,R)
        lambda_min = minimum(eigvals(Hess))
        if lambda_min < 0 && ~run_to_end
            jump = true
            return NaN, NaN, NaN, jump
        end

        if lambda_min < 0
            jump = true
            if ND_steps == 0
                r_prev = copy(r); e_prev = e
                simplex = ND_initialize(energy, r, step = max_step)
                no_improve = 0
            end

            r, e = simplex[1]
            if e < e_prev - no_improv_thr
                no_improve = 0
                e_prev = e, r_prev = copy(r)
            else
                no_improve += 1
            end

            if no_improv > max_no_improve
                println("Maximum number of steps without improvement reached during Nelder-Mead!")
                return r, e, lambda_min, jump
            end

            if ND_steps >= max_nd_steps
                println("Maximum number of total steps reached during Nelder-Mead!")
                return r, e, lambda_min, jump
            end

            simplex = ND_run_step(energy, simplex)
            r_error = sqrt(sum((simplex[end][1] - simplex[1][1].^2)))
            ND_steps += 1

        else
            ND_steps = 0
            delta_r = -inv(Hess)*grad(r,R)
            delta_r_norm = sqrt(sum(delta_r.^2))
            r_error = delta_r_norm
            if delta_r_norm > max_step
                delta_r = delta_r/delta_r_norm*max_step
                delta_r_norm = max_step
            end
            r += delta_r
            e = energy(r)
        end

        if r_error < min_error
            return r, e, lambda_min, jump
        end
    end

end

energy, grad, hess = Functions.get_functions(g = 0.01, kappa = 1, Delta = [0.002, 0])

function runs()
    for i in 1:155
        r, e, lambda_min, jump = sliding(energy, grad, hess, [-10., 0., -10., 0.], [-10., 0.3], 0.2;
            min_error = 1e-6, no_improv_thr = 1e-6, max_no_improve = 1000, max_nd_steps = 1000,
            run_to_end = false)
    end
end

@time begin
runs()
end

@time begin
runs()
end

#println(r," ", e, " ", lambda_min," ", jump)
