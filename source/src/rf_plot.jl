using RecipesBase

function gather_series(sol, mp, s)
    map(u->u.x[mp][s], sol.u)
end

function plot_number_series(sol, series; kwargs...)
    data = map(s->gather_series(sol,s...), series)
    plot(sol.t, data; kwargs...)
end

function resolve_mp_s_series(system_info, series)
    map(series) do (mp, s)
        (system_info[:multi_pools_info][:multi_pool_idx][mp],
         system_info[:species_info][:species_idxs][s])
    end
end

function plot_solution(; kwargs...)
    kw = Dict(kwargs)
    # TODO: write me
end

function plot_traces(traces, plt_sol_fn)
    n = length(traces)

    plts = map(traces |> enumerate) do (i, t)
        res = Gen.get_retval(t)
        plt_sol_fn(res[:sol], res[:systems_info], i)
    end

    plot(plts..., layout = (n, 1), size = (600, 400*n))
end
