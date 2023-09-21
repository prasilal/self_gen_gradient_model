##################
## Rules Library #
##################

function rule_tensor_produce(rule, _, reqs, allocated, _, produced, used, state, t, system)
    out = rule.output.x
    foreach_item(produced) do i,j
        if out[i][j] != nothing
            pos, amount = out[i][j]
            produced[i][j][pos] .= amount * t
        end
    end
end

function comp_diff_masks(out)
    masks = [fill(false, length(out[i])) for i in 1:length(out)]
    foreach_item(out) do i, j
        if out[i][j] != nothing && out[i][j] != 0
            masks[i][out[i][j]] = true
        end
    end
    masks
end

function rule_tensor_diffuse(rule, _, reqs, allocated, _, produced, used, state, t, system)
    out = rule.output.x
    masks = comp_diff_masks(out)
    foreach_item(reqs) do i,j
        if (reqs[i][j] != 0) & (!masks[i][j])
            k = allocated[i][j] |> ndims |> masked_laplacian
            mask_smooth!(produced[i][j], allocated[i][j], allocated[i][out[i][j]], LocalFilters.Kernel(k))
            produced[i][j] .*= reqs[i][j]*t
        end
    end
end

function mk_init_rate(reqs, allocated)
    ai, aj = 0, 0

    for i in 1:length(reqs)
        for j in 1:length(reqs[i])
            if reqs[i][j] != 0
                ai = i; aj = j
                break
            end
        end
        if ai != 0
            break
        end
    end

    fill(1.0, size(allocated[ai][aj]))
end

@inline function tensor_reactive_kinetics(reqs, allocated, r)
    rate = mk_init_rate(reqs, allocated)

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0
            @. rate *= allocated[i][j] ^ reqs[i][j]
        end
    end

    rate .* r
end

@inline function tensor_reactive_products(p, reqs, produced, used, products)
    foreach_item(produced) do i,j
        if products[i][j] != 0
            @. produced[i][j] += products[i][j] * p
        end
        if reqs[i][j] != 0
            @. used[i][j] += reqs[i][j] * p
        end
    end
end

function rule_tensor_reactive_kinetics(r, rule, _, reqs, allocated, _, produced, used, state, t, system)
    products = get_products_stochiometry(rule, PARTITION)
    p = tensor_reactive_kinetics(reqs, allocated, r) .* t
    tensor_reactive_products(p, reqs, produced, used, products)
    # println("tensor rk: p: $p t:$t products: $products produced: $produced uded: $used")
end

function rule_scaled_reactive_dynamics(scale_fn, args, r, rule, _, reqs, allocated, _,
                                       produced, used, state, t, system)
    products = get_products_stochiometry(rule, PARTITION)
    p = scale_fn(reactive_kinetics(reqs, allocated, r), t, args...)
    foreach_item(produced) do i,j
        pr = p * products[i][j]
        us = p * reqs[i][j]
        if  products[i][j] != 0
            produced[i][j] += pr
        end
        if reqs[i][j] != 0
            if allocated[i][j] > (us - pr)
                used[i][j] += us
            else
                used[i][j] += allocated[i][j]
            end
        end
    end
end

@def_fn_rule rule_scaled_reactive_dynamics_arg begin
    args = rule_args(rule)
    @add_call_rule_args rule_scaled_reactive_dynamics(args[1], args[3:end], args[2])
end

const rule_saturated_reactive_dynamics = partial(rule_scaled_reactive_dynamics, saturated, ())

@def_fn_rule rule_hill_dynamic begin
    products = get_products_stochiometry(rule, PARTITION)
    ps = 1.0
    coefs = rule.input.x
    foreach_item(allocated) do i,j
        if coefs[i][j] != 0
            if typeof(coefs[i][j]) <: Tuple
                if length(coefs[i][j]) == 3
                    (n, k, kA) = coefs[i][j]
                    if k < 0
                        ps *= hill_inh(allocated[i][j], n, -k, kA)
                    else
                        ps *= hill_act(allocated[i][j], n, k, kA)
                    end
                else
                    (n, m, k, kA) = coefs[i][j]
                    if k < 0
                        ps *= hill_inh(allocated[i][j], n, m, -k, kA)
                    else
                        ps *= hill_act(allocated[i][j], n, m, k, kA)
                    end
                end
            elseif typeof(coefs[i][j]) <: Number
                ps *= allocated[i][j] ^ coefs[i][j]
            end
        end
    end
    ps *= t
    foreach_item(produced) do i,j
        if products[i][j] != 0
            produced[i][j] += products[i][j]*ps
        end
    end
end

function rule_inject(rule)
    products = get_products_stochiometry(rule, PARTITION)
    d = Dict()

    foreach_item(products) do i,j
        if products[i][j] != 0
            prep = d[(i,j)] = []
            foreach(products[i][j]) do par
                k = 10
                if length(par) == 2
                    _, amount = par
                else
                    _, amount, k = par
                end
                k = amount > 0 ? k : -k
                b = amount / (2*k)
                push!(prep, [0, k, b])
            end
        end
    end

    d
end

function rule_inject(rule, _, reqs, allocated, _, produced, used, state, t, system)
    products = get_products_stochiometry(rule, PARTITION)
    curr_t = system.cache[:t]

    foreach_item(produced) do i,j
        if products[i][j] != 0
            times = products[i][j]
            foreach(times, state[(i,j)]) do par, prep
                dc = (curr_t < par[1] - prep[3] ||
                      ((curr_t > par[1] + prep[3]) && (prep[1] == par[2]))) ?
                    0 : prep[2] * t
                if dc != 0
                    if par[2] >= 0
                        dc = prep[1] + dc <= par[2] ? dc : par[2] - prep[1]
                    else
                        dc = prep[1] + dc >= par[2] ? dc : par[2] - prep[1]
                    end
                    produced[i][j] += dc

                    prep[1] += dc
                end
            end
        end
    end
end

# constant production
@def_fn_rule rule_const_prod begin
    products = get_products_stochiometry(rule, PARTITION)

    foreach_item(produced) do i, j
        if products[i][j] != 0
            produced[i][j] += products[i][j]*t
        end
    end
end

# returns either multi pool on given index or return tmp mpool from cache
get_sys_mpool(system, idx :: T) where {T <: Integer} = get_mpool_cont(system.multi_pools, idx)
get_sys_mpool(system, idx :: T) where {T <: Symbol} = system.cache[:multi_pools_tmp][idx]

@def_fn_rule rule_multi_pool_copy_preproc_args begin
    (src_idx, dst_idx, f, for_all) = rule_args(rule)

    src = get_sys_mpool(system, src_idx)
    dst = get_sys_mpool(system, dst_idx)

    if for_all
        foreach_item(src) do i, j
            dst[i][j] = f(src[i][j])
        end
    else
        foreach_item(src) do i, j
            if reqs[i][j] != 0
                dst[i][j] = f(src[i][j])
            end
        end
    end
end

@def_fn_rule rule_multi_pool_new_tmp_args begin
    (src_idx, dst_name, f) = rule_args(rule)

    system.cache[:multi_pools_tmp][dst_name] = f(system.multi_pools.x[src_idx])
end

@def_fn_rule rule_multi_pool_op_args begin
    (lhs_idx, op, rhs_op1_idx, rhs_op2_idx, for_all) = rule_args(rule)

    lhs = get_sys_mpool(system, lhs_idx)
    rhs_op1 = get_sys_mpool(system, rhs_op1_idx)
    rhs_op2 = get_sys_mpool(system, rhs_op2_idx)

    if for_all
        foreach_item(lhs) do i, j
            lhs[i][j] = op(rhs_op1[i][j], rhs_op2[i][j])
        end
    else
        foreach_item(lhs) do i, j
            if reqs[i][j] != 0
                lhs[i][j] = op(rhs_op1[i][j], rhs_op2[i][j])
            end
        end
    end
end

@def_fn_rule rule_cond_action_args begin
    cond, cond_args, act, act_args = rule_args(rule)

    if (@add_call_rule_args cond(cond_args...)) == true
        @add_call_rule_args act(act_args...)
    end
end

@def_fn_rule cond_any begin
    res = false
    foreach_item(reqs) do i,j
        res = res || pred(reqs[i][j], allocated[i][j])
        return res
    end
end pred

@def_fn_rule cond_all begin
    res = true
    foreach_item(reqs) do i,j
        res = res && pred(reqs[i][j], allocated[i][j])
        return !res
    end
end pred

@def_fn_rule cond_min_react begin
    reactive_kinetics(reqs, allocated, r) >= min
end r min

cond_min_amount(r :: TR, a :: TA) where {TR <: Real, TA <: Real} = a >= r
cond_min_amount(r :: TR, a :: Agents) where {TR <: Real} = length(a) >= r

@def_fn_rule act_for_nonzero begin
    products = get_products_stochiometry(rule, PARTITION)
    foreach_item(products) do i,j
        if products[i][j] != 0
            f(fargs(t)..., produced, i, j)
        end
    end
end f fargs

gen_evt(etp, eargs, produced, i, j) = push!(produced[i][j], gen_ev(etp, eargs))

###########
## WRAPPERS

function make_diffuse_rule(rin, rout, label)
    make_gen_rule(
        Tensor,
        Function,
        rin,
        rout,
        rule_tensor_diffuse,
        label
    )
end

function make_point_production_rule(rin, rout, label)
    make_gen_rule(
        Tensor,
        Function,
        rin,
        rout,
        rule_tensor_produce,
        label
    )
end
