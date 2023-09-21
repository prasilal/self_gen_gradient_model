macro def_fn_rule(name, body, args...)
    return :(function $name($(args...),
                            rule, state_desc, reqs, allocated, allocated_per_mpool,
                            produced, used, state, t, system)
             $body
             end)
end

const rule_args_names = [:rule, :state_desc, :reqs, :allocated, :allocated_per_mpool,
                         :produced, :used, :state, :t, :system]

macro add_call_rule_args(f)
    append!(f.args, rule_args_names)
    f
end

macro def_apply_rule!(tp, tp_cond, body)
    return :(function apply_rule!(rule :: $tp, state_desc, reqs,
                                  allocated, allocated_per_mpool,
                                  produced, used, state, t, system) where {$tp_cond}
             $body
             end)
end

function make_view(m, rows, columns)
    map(sm -> view(sm, rows, columns), m)
end

function comp_ratios(used, allocated, reqs, r)
    used .= allocated .* r
    used ./ epsilonize(reqs)
end

function k_products_ratio(rule, t)
    k = get_k(rule)
    products = get_products_stochiometry(rule, PARTITION)
    k, products, min(k == 0 ? 1 : t / k, 1.)
end

function update_produced!(ratios, reqs, produced, products)
    min_ratio = minimum(nonzero_mins(ratios', reqs'))
    foreach((pr, ps) -> pr .+= ps .* min_ratio, produced, products)
end

function apply_rule!(rule :: ReactiveRule{T}, _, reqs, allocated, _,
                     produced, used, _ , t, _) where T <: AbstractFloat
    _, products, r = k_products_ratio(rule, t)
    ratios = map((us, al, rq) -> comp_ratios(us, al, rq, r), used, allocated, reqs)
    # println("apply_rule!", r, " ", allocated, " ", reqs, " ", ratios)
    update_produced!(ratios, reqs, produced, products)
end

function comp_ratios(allocated, reqs)
    map((al, rs) -> floor.(al ./ epsilonize(rs)), allocated, reqs)
end

function apply_rule!(rule :: ReactiveRule{T}, _, reqs, allocated, _,
                     produced, used, state, t, _) where T <: Integer
    k, products, _ = k_products_ratio(rule, t)
    int_queue = state

    ratio = comp_ratios(allocated, reqs)
    m = minimum(nonzero_mins(ratio', reqs'))
    fully_allocated = map(rs -> rs .* m, reqs)

    if m != 0
        push!(int_queue, [fully_allocated, 0])
        foreach((u, a) -> u .= a, used, fully_allocated)
    end

    foreach(s -> s[2] += t, int_queue)
    while (length(int_queue) > 0) && (int_queue[1][2] >= k)
        alloc, _ = popfirst!(int_queue)
        ratios = comp_ratios(alloc, reqs)
        update_produced!(ratios, reqs, produced, products)
    end
end

function apply_rule!(rule :: ReactiveRule{T}, state_desc, reqs, allocated, allocated_per_mpool,
                     produced, used, state, t, system) where T <: MemoryInt
    used_tmp = deepcopy(used)
    foreach((u, a) -> u .= a, used, allocated)
    remains = state[2]

    if remains != nothing
        current_allocated = map((a, r) -> a .+ r, allocated, remains)
    else
        current_allocated = allocated
    end

    apply_rule!(convert(ReactiveRule{Integer}, r), state_desc, reqs, current_allocated,
                allocated_per_mpool,produced, used_tmp, state[1], t, system)

    state[2] = map((c, f) -> c .- f, current_allocated, used_tmp)
end

# https://en.wikipedia.org/wiki/Autocatalysis
#
# k_{+}[A]^{\alpha }[B]^{\beta }=k_{-}[S]^{\sigma }[T]^{\tau }\,
# 
# {d \over dt}[A]=-\alpha k_{+}[A]^{\alpha }[B]^{\beta }+\alpha k_{-}[S]^{\sigma }[T]^{\tau }\,
# {\displaystyle {d \over dt}[B]=-\beta k_{+}[A]^{\alpha }[B]^{\beta }+\beta k_{-}[S]^{\sigma }[T]^{\tau }\,}{d \over dt}[B]=-\beta k_{+}[A]^{\alpha }[B]^{\beta }+\beta k_{-}[S]^{\sigma }[T]^{\tau }\,
# {\displaystyle {d \over dt}[S]=\sigma k_{+}[A]^{\alpha }[B]^{\beta }-\sigma k_{-}[S]^{\sigma }[T]^{\tau }\,}{d \over dt}[S]=\sigma k_{+}[A]^{\alpha }[B]^{\beta }-\sigma k_{-}[S]^{\sigma }[T]^{\tau }\,
# {\displaystyle {d \over dt}[T]=\tau k_{+}[A]^{\alpha }[B]^{\beta }-\tau k_{-}[S]^{\sigma }[T]^{\tau }\,}{d \over dt}[T]=\tau k_{+}[A]^{\alpha }[B]^{\beta }-\tau k_{-}[S]^{\sigma }[T]^{\tau }\,
@inline function reactive_kinetics(reqs, allocated, r)
    res = r

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0
            res *= allocated[i][j] ^ reqs[i][j]
        end
    end

    res

    # sub_prods = map((rs, al) -> al .^ rs, reqs, allocated)
    # reduce((sp, x) -> sp * prod(x), sub_prods, init = 1) * r
end

@inline function reactive_products(p, reqs, produced, used, products)
    foreach_item(products) do i,j
        if products[i][j] != 0
            produced[i][j] += products[i][j] * p
        end
        if reqs[i][j] != 0
            used[i][j] += reqs[i][j] * p
        end
    end

    # foreach((prod, ps) -> prod .+= ps .* p, produced, products)
    # foreach((u, rs) -> u .+= rs .* p, used, reqs)
end

function apply_rule!(rule :: ReactiveRule{T}, _, reqs, allocated, _,
                     produced, used, state, t, _) where T <: MassActionReal
    r, products, _ = k_products_ratio(rule, t)
    p = reactive_kinetics(reqs, allocated, r) * t
    reactive_products(p, reqs, produced, used, products)
    # sub_prods = map((rs, al) -> al .^ rs, reqs, allocated)
    # p = reduce((sp, x) -> sp * prod(x), sub_prods, init = 1) * r * t
    # foreach((prod, ps) -> prod .+= ps .* p, produced, products)
    # foreach((u, rs) -> u .+= (rs .!= 0) .* p, used, reqs)
end

function apply_rule!(rule :: ReactiveRuleGroup, state_desc, reqs, allocated, allocated_per_mpool,
                     produced, used, state, t, system)
    foreach((r, s) -> apply_rule!(r, state_desc, get_reactants_stochiometry(r, PARTITION) |> collect,
                                  allocated, allocated_per_mpool, produced, used, s, t, system),
            rule.rules, state)
end

rule_fn(r :: GeneralReactiveRule{T,S}) where {T, S <: Function} = rule.k

function apply_rule!(rule :: GeneralReactiveRule{T,S}, state_desc, reqs, allocated, allocated_per_mpool,
                     produced, used, state, t, system) where {T, S <: Function}
    rule.k(rule, state_desc, reqs, allocated, allocated_per_mpool, produced, used, state, t, system)
end

rule_fn(r :: GeneralReactiveRule{T,S}) where {T, S <: FnArgs} = r.k[1]
rule_args(r :: GeneralReactiveRule{T,S}) where {T, S <: FnArgs} = r.k[2]

function apply_rule!(rule :: GeneralReactiveRule{T,S}, state_desc, reqs, allocated, allocated_per_mpool,
                     produced, used, state, t, system) where {T, S <: FnArgs}
    rule.k[1](rule, state_desc, reqs, allocated, allocated_per_mpool, produced, used, state, t, system)
end

@def_apply_rule! TimeDelayedRule T begin
    cache = get(state, :delay_cache, nothing)

    if cache == nothing || rule.T == 0
        h_alloc = allocated
        h_alloc_pm = allocated_per_mpool
    else
        (_, h_alloc, h_alloc_pm) = cache[1]
    end

    apply_rule!(rule.rule, state_desc, reqs, h_alloc, h_alloc_pm, produced, used,
                get(state, :state, nothing), t, system)

    if rule.T != 0
        if cache == nothing
            cache = state[:delay_cache] = []
            state[:t] = 0.0
        else
            dt = state[:t] + t

            if dt > rule.T
                c = popfirst!(cache)
                state[:t] -= c[1]
            end
        end

        state[:t] += t
        push!(cache, (t, deepcopy(allocated), deepcopy(allocated_per_mpool)))
    end
end

function update_produced!(pool :: Pool{T,N}, produced, production_weights, used, ratio, _) where {T <: Real, N}
    # println("update_produced!: ", produced, production_weights, used, ratio)
    pool.x .+= sum_by_row(produced .* production_weights .- used .* ratio)
    # println("pool: ", pool, "\n")
end

function update_produced!(pool :: Pool{T,N}, produced, production_weights, used, ratio, _) where {T <: Condition, N}
end

function update_produced!(pool :: Pool{T,N}, produced, production_weights, used, ratio, _) where {T <: Agents, N}
    for (i, pi) in enumerate(pool)
        del_agents!(pi, vcat(used[:,i]...))
        p = map((pr, pw) -> splice!(pr, 1:min(pw, length(pr))),
                produced[:,i], production_weights[:,i])
        append!(pi, vcat(p...))
    end
end

function update_produced!(pool :: Pool{Array{T, NN}, N}, produced, production_weights, used, ratio, _) where {T <: Real, N, NN}
    foreach(pool |> enumerate) do (i, p)
        p .+= reduce(+, produced[:,i] .* production_weights[:,i]) .- reduce(+, used[:,i] .* ratio[:, i])
    end
end

function comp_pool_ratios(pool :: Pool{T,N}, allocated_per_mpool, allocated) where {T <: Agents, N}
    length.(allocated_per_mpool) ./ (length.(allocated) |> epsilonize)
end

# TODO: can we optimize this ?
function comp_pool_ratios(pool :: Pool{Array{T, NN}, N}, allocated_per_mpool, allocated) where {T <: Real, NN, N}
    sum.(allocated_per_mpool) ./ (sum.(allocated) |> epsilonize)
end

function comp_pool_ratios(pool :: Pool{T,N}, allocated_per_mpool, allocated) where {T,N}
    # println("comp_pool_ratios: $pool $allocated $allocated_per_mpool")
    allocated_per_mpool ./ epsilonize(allocated)
end

function preproc_prod_weights(pool :: Pool{T,N}, produced, ws) where {T <: Agents,N}
    ls = length.(produced)
    pws = floor.(ls ./ epsilonize(ws))
    convert.(Int, pws)
end

function preproc_prod_weights(pool :: Pool{T,N}, produced, ws) where {T,N}
    # println("preproc_prod_weights: $pool $produced $ws")
    ws
end

# Apply all rules in system given allocated species per rule and per multi pool
# Dived produced species into multi pools regarding production weights
function apply!(mpools, rules_by_state, reqs, allocated, allocated_per_mpool,
                production_weights, rules_state, t, system; rules_by_state_msk = nothing)
    produced = empty_allocation(mpools.x[1], reqs)
    used = empty_allocation(mpools.x[1], reqs)

    # TODO: partition it by rule type and do it more vectorized
    for (i, rs) in enumerate(rules_by_state)
        if rules_by_state_msk == nothing || rules_by_state_msk[i]
            rule, state_desc = rs
            state_desc = Dict(:state_desc => state_desc, :alloc_type => system.cache[:rows_allocators][i])
            apply_rule!(rule, state_desc, make_view(reqs, i, :), make_view(allocated, i, :),
                        map(mp -> make_view(mp, i, :), allocated_per_mpool),
                        make_view(produced, i, :), make_view(used, i, :), rules_state[i], t, system)
        end
    end

    # alloc_eps = epsilonize(allocated)
    foreach((mpool, alloc_pp, prod_ws) ->
            foreach(
                (pool, prod, pws, us, al, alpp) ->
                  update_produced!(pool, prod, preproc_prod_weights(pool, prod, pws), us,
                    comp_pool_ratios(pool, alpp, al), system),
                mpool.x, produced, prod_ws, used, allocated, alloc_pp),
            mpools.x, allocated_per_mpool, production_weights)
end
