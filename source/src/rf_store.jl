using DataFrames

function flatten_series(sol, series)
    n = length(sol.t)
    d = Array{Any}(undef, n * length(series))
    offset = 1

    for s in series
        d[offset:(offset+n-1)] .= gather_series(sol, s...)
        offset += n
    end

    d
end

function flatten_sol(series, sol, systems_info, idx)
    flatten_series(sol,
                   resolve_mp_s_series(get_system_info(systems_info, :default),
                                       series))
end

function store_traces_to_df(df, traces, flatten_sol_fn)
    sols = map(traces |> enumerate) do (i, t)
        res = Gen.get_retval(t)
        flatten_sol_fn(res[:sol], res[:systems_info], i)
    end

    tmp_df = @> sols vec_of_vec_to_mtx(dims = 2) DataFrame

    append!(df, tmp_df)
end
