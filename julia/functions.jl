module Functions

function tensor_dot(v1, v2)
    M = []
    for (i, v) in enumerate(v1)
        if i == 1
            M = v*hcat(v2)
        else
            M = [M; v*hcat(v2)]
        end
    end
    return M
end

function get_functions(; g, kappa, Delta)
    C = 0.25/kappa

    function energy(r, R)
        r1 = r[1:2]; r2 = r[3:4]
        u1 = r1 - (R - Delta/2)
        u2 = r2 - (R + Delta/2)
        return -1/(1 + sum(r1.^2)/2) - 1/(1 + sum(r2.^2)/2) + 1/2*C/(1-g^2)*sum(u1.^2 + u2.^2 - 2*g*u1.*u2)
    end

    function grad(r,R)
        r1 = r[1:2]; r2 = r[3:4]
        u1 = r1 - (R - Delta/2)
        u2 = r2 - (R + Delta/2)
        dr1_e = r1/(1 + sum(r1.^2)/2)^2 + C/(1-g^2)*(u1 - g*u2)
        dr2_e = r2/(1 + sum(r2.^2)/2)^2 + C/(1-g^2)*(u2 - g*u1)
        return vcat(dr1_e, dr2_e)
    end

    function hess(r,R)
        r1 = r[1:2]; r2 = r[3:4]
        u1 = r1 - (R - Delta/2)
        u2 = r2 - (R + Delta/2)
        diag = [1 0; 0 1]

        denom_r1 = 1 + sum(r1.^2)/2; denom_r2 = denom_r1 = 1 + sum(r2.^2)/2
        alpha = C/(1-g^2)

        dr1r1_e = diag/denom_r1^2 - 2*tensor_dot(r1, transpose(r1))/denom_r1^3 + alpha*diag
        dr1r2_e = -g*alpha*diag
        dr2r2_e = diag/denom_r2^2 - 2*tensor_dot(r2, transpose(r2))/denom_r2^3 + alpha*diag
        return [dr1r1_e dr1r2_e; dr1r2_e dr2r2_e]
    end

    return energy, grad, hess

end

end
