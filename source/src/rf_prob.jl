import Gen

Gen.@gen function rf_prob_model(kwargs)
    @assert(haskey(kwargs, :sim_fn))

    kw = copy(kwargs)

    trace_step = get(kw, :meas_step, 1)
    noise = get(kw, :meas_noise, 0.01)
    mdist = get(kw, :meas_dist, Gen.normal)
    sim_fn = get(kw, :sim_fn, nothing)

    dps = get(kw, :dist_params, [])
    for (ks, v) in dps
        f, args = v
        foreach(k -> kw[k] = Gen.@trace(f(args...), :params => k) , ks)
    end

    foreach(k -> delete!(kw, k),
            [:dist_params, :meas_step, :meas_noise, :meas_dist, :sim_fn])

    res = sim_fn(;kw...)
    sol = res[:sol]

    for i in 1:trace_step:length(sol.u)
        u = sol.u[i].x

        for j in 1:length(u)
            for k in 1:length(u[j])
                Gen.@trace(mdist(u[j][k], noise), :sol => (i, j, k))
            end
        end
    end

    res
end

function rf_prob_model_gen(;model_kwargs = Dict(), num_runs = 100)
    traces = []

    for _ in 1:num_runs
        (trace, _) = Gen.generate(rf_prob_model, (model_kwargs,));
        push!(traces, trace)
    end

    traces
end

function get_trace_data(trace)
    sol = Gen.get_retval(trace)[:sol]
    kw = Gen.get_args(trace)
    t_step = trace_step = get(kw, :meas_step, 1)
    data = []
    for i in 1:t_step:length(sol.u)
        u = sol.u[i].x

        for j in 1:length(u)
            for k in 1:length(u[j])
                push!(data, (i,j,k,trace[:sol=>(i,j,k)]))
            end
        end
    end
    data
end

function get_trace_params(trace)
    c = Gen.get_choices(trace)
    pc = Gen.get_submap(c, :params)
    ps = Gen.get_leaf_nodes(pc.trie)
    params = Dict()
    for k in keys(ps)
        params[k] = ps[k].subtrace_or_retval
    end

    params
end

function do_inference_agent_model(kw, params, measurements, amount_of_computation::Int)
    observations = Gen.choicemap()

    for (k, v) in collect(params)
        observations[k] = v
    end

    for m in measurements
        observations[:sol => m[:addr]] = m[:val]
    end

    (trace, _) = Gen.importance_resampling(rf_prob_model, (kw,), observations, amount_of_computation)

    return trace
end
