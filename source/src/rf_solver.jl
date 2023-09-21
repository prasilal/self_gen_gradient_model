####################################
### Simple constant dt step solver #
####################################

sol_default_realloc(::Val{:s}, _, _) = []
sol_default_realloc(::Val{:t}, _, _) = []
sol_default_realloc(::Val{:u}, _, _) = []

const VAL_S = Val{:s}()
const VAL_T = Val{:t}()
const VAL_U = Val{:u}()

const SOL_DEFAULT_PARAMS = Dict(
    :realloc => sol_default_realloc,
    :realloc_params => nothing
)

function init_solver(::Type{Val{:simple}}; kwargs...)
    kwargs = Dict(kwargs)
    params = get(kwargs, :params, SOL_DEFAULT_PARAMS)
    realloc = params[:realloc]
    realloc_params = params[:realloc_params]

    Solution(Val{:simple}(),
             realloc(VAL_U, realloc_params, nothing),
             realloc(VAL_T, realloc_params, nothing),
             realloc(VAL_S, realloc_params, nothing),
             kwargs[:system],
             Dict(
                 :ts => get(kwargs, :ts, 0.1),
                 :tspan => get(kwargs, :tspan, (0., 1.))
             ),
             params)
end

function range_intersection(r1, r2)
    if r1.stop < r2.start || r2.stop < r1.start
        nothing
    else
        mi = max(r1.start, r2.start)
        ma = min(r1.stop, r2.stop)
        mi:ma
    end
end

function update_tspan!(sol :: Solution{Val{:simple}, Tu, Tt, Ts, Tsystem, Tsolver, Tparams}, new_tspan) where {Tu, Tt, Ts, Tsystem, Tsolver, Tparams}
    old_tspan = sol.solver[:tspan]
    realloc = params[:realloc]
    realloc_params = paramss[:realloc_params]

    if old_tspan == new_tspan
        tspan = new_tspan
        sol.u = realloc(VAL_U, realloc_params, sol.u)
        sol.t = realloc(VAL_U, realloc_params, sol.t)
        sol.s = realloc(VAL_U, realloc_params, sol.s)
    else
        tspan_int = range_intersection(old_tspan, new_tspan)
        if old_tspan.start < new_tspan.start && new_tspan != tspan_int
            tspan = old_tspan.stop:new_tspan.stop
        else # we do not support time traveling yet ;)
            tspan = new_tspan
            sol.u = realloc(VAL_U, realloc_params, sol.u)
            sol.t = realloc(VAL_U, realloc_params, sol.t)
            sol.s = realloc(VAL_U, realloc_params, sol.s)
        end
    end
    sol.solver[:tspan] = tspan
end

function save_sol!(sol, system, t; save_states = true)
    push!(sol.u, deepcopy(system.multi_pools))
    if save_states
        push!(sol.s, deepcopy(system.cache[:rules_state]))
    end
    push!(sol.t, t)
end

function solve!(sol :: Solution{Val{:simple}, Tu, Tt, Ts, Tsystem, Tsolver, Tparams}; kwargs...) where {Tu, Tt, Ts, Tsystem, Tsolver, Tparams}
    kwargs = Dict(kwargs)
    dt = haskey(kwargs, :ts) ? kwargs[:ts] : sol.solver[:ts]
    save_states = get(kwargs, :save_states, true)
    update_last = get(kwargs, :update_last, true)

    params = get(kwargs, :params, SOL_DEFAULT_PARAMS)
    realloc = params[:realloc]
    realloc_params = params[:realloc_params]

    if haskey(kwargs, :continue) && kwargs[:continue]
        if haskey(kwargs, :tspan)
            update_tspan!(sol, kwargs[:tspan])
        end
    elseif !isempty(sol.u)
        sol.u = realloc(VAL_U, realloc_params, sol.u)
        sol.t = realloc(VAL_U, realloc_params, sol.t)
        sol.s = realloc(VAL_U, realloc_params, sol.s)
    end

    (start, stop) = sol.solver[:tspan]
    system = sol.system
    for t in range(start, stop, step = dt)
        save_sol!(sol, system, t, save_states = save_states)
        system.cache[:t] = t
        if t != stop || update_last
            rf_step!(system, dt)
        end
    end

    sol
end

# @TODO: remove deprecated
const solve = solve!

#####################
### Advanced solver #
#####################

is_cont(p :: Pool{T,N}) where {T <: Real, N} = true
is_cont(p :: Pool{T,N} where {T <: AbstractArray{TT, NN} where {TT <: Real, NN}}) where N = true
is_cont(p :: Pool{T,N}) where {T, N} = false

get_cont_idxs(system) = findall(is_cont, system.multi_pools.x[1].x)

function make_cont_pools_mask(system, idxs)
    msk = fill(false, length(system.multi_pools.x[1].x))
    msk[idxs] .= true
    msk
end

isntzero(x) = x != 0

function make_cont_state_rules_mask(system, idxs)
    map(system.cache[:rules_by_state]) do (r, _)
        reqs = get_reactants_stochiometry(r, PARTITION)
        prod = get_products_stochiometry(r, PARTITION)
        any(reqs[idxs]) do r
            any(isntzero, r)
        end || any(prod[idxs]) do p
            any(isntzero, p)
        end
    end
end

push_pool!(a, p :: Pool{T,N}) where {T <: Real, N} = push!(a, p)
push_pool!(a, p :: Pool{T,N} where {T <: AbstractArray{TT, NN} where {TT <: Real, NN}}) where N = push!(a, BigArrayPartition(p.x))

function view_cont_parts(multi_pools, idxs)
    cpart = []
    foreach(multi_pools.x) do mp
        foreach(idxs) do i
            push_pool!(cpart, mp.x[i])
        end
    end
    BigArrayPartition(cpart)
end

const T = 1
const MPS = 2
const RSTS = 3
const CV = 4
const DF = 5

@inline comp_df(v_start, v_end, dt) = convert(Vector, v_end - v_start) ./ dt

# TODO: What to do with transitions ? Transit only continuous species ?
function sys_df(u, p :: Solution{Val{:advanced}, Tu, Tt, Ts, Tsystem, Tsolver, Tparams}, t) where {Tu, Tt, Ts, Tsystem, Tsolver, Tparams}
    val_cache = get(p.solver, :val_cache, [])
    res = searchsorted(val_cache, t, by = first)
    l = res.start
    h = res.stop

    df = nothing

    if l == h
        if l == 1
            df = val_cache[l][DF]
        else
            (t_start, _, _, v_start, _) = val_cache[l - 1]
            (_, mps, rst, v_end, _) = val_cache[l]
            df = comp_df(v_start, v_end, t - t_start)
            val_cache[l] = (t, mps, rst, v_end, df)
        end
    else
        system = p.system

        rmask = get(p.solver, :rules_msk, [])
        pmask = get(p.solver, :pools_msk, [])
        cont_idxs = get(p.solver, :cont_idxs, [])
        (t_start, mps, rsts, v_start, _) = val_cache[h]

        system.multi_pools = deepcopy(mps)
        setcache!(system, :rules_state, deepcopy(rsts)) # system.cache[:rules_state] = deepcopy(rsts)
        system.cache[:t] = t
        rf_step!(system, t - t_start, rules_msk = rmask, pools_mask = pmask)

        mps_end = deepcopy(system.multi_pools)
        v_end = view_cont_parts(mps_end, cont_idxs)
        df = comp_df(v_start, v_end, t - t_start)

        p.solver[:val_cache] = val_cache = val_cache[1:h]
        push!(val_cache, (t, mps_end, deepcopy(system.cache[:rules_state]), v_end, df))
    end

    df
end

const ODE_INTEGRATOR_KWARGS = Set(
    [:adaptive, :abstol, :reltol, :dtmax, :dtmin, :dense, :saveat, :save_idxs,
     :tstops, :d_discontinuities, :save_everystep, :save_on, :save_start, :save_end,
     :initialize_save, :alg_hints, :maxiters, :callback, :isoutofdomain, :unstable_check,
     :verbose, :merge_callbacks]
)

function init_solver(::Type{Val{:advanced}}; kwargs...)
    kwargs = Dict(kwargs)
    params = get(kwargs, :params, SOL_DEFAULT_PARAMS)
    realloc = params[:realloc]
    realloc_params = params[:realloc_params]

    system = kwargs[:system]
    tspan = get(kwargs, :tspan, (0., 1.))
    idxs = get_cont_idxs(system)

    mps_start = deepcopy(system.multi_pools)
    rsts_start = deepcopy(system.cache[:rules_state])
    v_start = view_cont_parts(mps_start, idxs)

    rmask = make_cont_state_rules_mask(system, idxs)
    pmask = make_cont_pools_mask(system, idxs)

    dt = eps(Float16) # TODO: better guess dt
    system.cache[:t] = tspan[1]
    rf_step!(system, dt, rules_msk = rmask, pools_mask = pmask)
    v_end = view_cont_parts(system.multi_pools, idxs)
    df = comp_df(v_start, v_end, dt)

    system.multi_pools = deepcopy(mps_start)
    setcache!(system, :rules_state, deepcopy(rsts_start)) # system.cache[:rules_state] = deepcopy(rsts_start)

    solver = Dict(
        :ts => get(kwargs, :ts, get(kwargs, :dt, 0.1)),
        :tspan => tspan,
        :cont_idxs => idxs,
        :rules_msk => rmask,
        :pools_msk => pmask,
        :val_cache => Any[(tspan[1], mps_start, rsts_start, v_start, df)]
    )

    sol = Solution(Val{:advanced}(),
                   realloc(VAL_U, realloc_params, nothing),
                   realloc(VAL_T, realloc_params, nothing),
                   realloc(VAL_S, realloc_params, nothing),
                   system, solver, params)

    tp = RecursiveArrayTools.recursive_unitless_bottom_eltype(v_start)
    solver[:ode_problem] = prob = ODEProblem(sys_df, convert(Vector{tp}, v_start), tspan, sol)
    alg = get(kwargs, :alg, AutoTsit5(Rosenbrock23()))
    solver[:ode_integrator] = DifferentialEquations.init(prob, alg;
                                                         dt = solver[:ts],
                                                         select_keys(kwargs, ODE_INTEGRATOR_KWARGS)...)

    sol
end

function save_sol_from_val_cache(sol; save_states = true)
    for (t, mps, sts, _, _) in sol.solver[:val_cache]
        if length(sol.t) == 0 || sol.t[end] != t
            push!(sol.u, mps)
            if save_states
                push!(sol.s, sts)
            end
            push!(sol.t, t)
        end
    end
end

function solve!(sol :: Solution{Val{:advanced}, Tu, Tt, Ts, Tsystem, Tsolver, Tparams}; kwargs...) where {Tu, Tt, Ts, Tsystem, Tsolver, Tparams}
    kwargs = Dict(kwargs)
    dt = haskey(kwargs, :ts) ? kwargs[:ts] : sol.solver[:ts]
    save_states = get(kwargs, :save_states, true)
    update_last = get(kwargs, :update_last, true)

    (start, stop) = sol.solver[:tspan]
    system = sol.system
    integrator = sol.solver[:ode_integrator]
    rmask = sol.solver[:rules_msk]
    if any(identity, rmask)
        for _ in integrator
            set_proposed_dt!(integrator, dt)
            save_sol_from_val_cache(sol, save_states = save_states)
            sol.solver[:val_cache] = [last(sol.solver[:val_cache])]
            if check_error(integrator) != :Success
                println("solve! error: $(check_error(sol.solver[:ode_integrator]))")
                break
            end
        end
    else
        for t in range(start, stop, step = dt)
            save_sol!(sol, system, t, save_states = save_states)
            system.cache[:t] = t
            if t != stop || update_last
                rf_step!(system, dt)
            end
        end
    end

    sol
end
