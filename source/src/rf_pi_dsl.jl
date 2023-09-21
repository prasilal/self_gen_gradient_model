#=
include("rf.jl")

partial(f, args...) = (next_args...) -> f(args..., next_args...)
struct MassActionContinuous <: Real end
=#

macro with_new_ctx(body)
    return :(push_ctx();
             try
               $(esc(body))
             finally
               pop_ctx();
             end)
end

function append_o_push!(a, v)
    if typeof(v) == Expr && v.head == :tuple
        append!(a,v.args)
    else
        push!(a,v)
    end
end

function parse_rate_body(expr)
    if typeof(expr) != Expr || expr.head != :call
        return esc(expr)
    end

    args = map(ex -> parse_rate_body(ex), expr.args)
    as = reduce(append_o_push!, args[2:end], init = [])
    Expr(:tuple, args[1], Expr(:tuple, as...))
end

macro with_rate(expr)
    body = parse_rate_body(expr)
    return Expr(:tuple, :(:->), body)
end

iscol(x :: Array) = true
iscol(x :: Tuple) = true
iscol(x :: Set) = true
iscol(x :: Dict) = true
iscol(x) = false

# current context
CUR_CTX = nothing

CTX_STACK = []

# MASK DSL

any_p(_) = true
any_item_p(x) = x !== nothing
eq_p(x) = y -> x == y
pred_p(f) = y -> f(y)

function rep(pred)
    (handler, state, input) -> begin
        while true
            if handler(state, input)
                return true
            end
            if !(input |> first |> pred)
                return false
            end
            if isempty(input)
                return false
            end
            input = tail(input)
        end
    end
end

const rep_any = rep(any_p)

term(handler, state, input) = isempty(input)
beg(handler, state, input) = length(state[:input]) == length(input) && handler(state,input)

mcont(state, input) = true
meq(x) = (handler, state, input) -> x == first(input) && handler(state, tail(input))
mitem(p) = (handler, state, input) -> (input |> first |> p) && handler(state, tail(input))
menum(xs) = (xs = Set(xs); (handler, state, input) -> first(input) in xs && handler(state, tail(input)))

function compile_mask(mask)
    fns = map(x -> typeof(x) <: Function ? x : meq(x), mask)
    reduce((mf, f) -> partial(f, mf), reverse(fns), init = mcont)
end

function colify(x)
    iscol(x) ? x : (x,)
end

function match_mask(mask, input)
    mfn = typeof(mask) <: Function ? mask : mask |> colify |> compile_mask
    input = colify(input)
    in = @lazy input
    for _ in range(1, length=length(input) + 1)
        if mfn(Dict(:input => input), in)
            return true
        end
        in = tail(in)
    end
    return false
end

mmask(msk) = (handler, state, input) -> first(input) != nothing && match_mask(msk, first(input)) && handler(state, tail(input))

const REAM = (rep_any,)

#=
mask_fn = compile_mask((1,rep(eq_p(2)),3))
mask_fn(Dict, @lazy [1,2,2,2,3])
match_mask((1,rep(eq_p(2)),3), (1,1,3,4,3))
match_mask((1,rep(eq_p(2)),3,term), (1,1,2,3,4))
match_mask((beg,1,rep(eq_p(2)),3), (1,2,3,4))
match_mask((rep_any, :b, term), ())

match_mask((rep_any,), ())

match_mask((beg, :species, mmask((:default,)), rep_any, :x, mitem(any_item_p), term),
(:species, (:default, Float64), :x, 1))

match_mask((beg, :species, mmask((:default,)), rep_any, :x, mitem(any_item_p), term),
:aa)

match_mask((beg, :species, mmask((:default,)), rep_any, :x, mitem(any_item_p), term),
(:species,))

match_mask()
=#

# MODEL DSL

function new_ctx()
    global CUR_CTX = Dict(:species => Dict(),
                          :multi_pools => Dict(),
                          :rules => Dict(),
                          :rules_applications => [],
                          :states => Dict(),
                          :trans_mtxs => Dict(),
                          :transitions => Dict(),
                          :systems => Dict())
end

function push_ctx()
    push!(CTX_STACK, CUR_CTX)
    new_ctx()
end

function pop_ctx()
    global CUR_CTX = pop!(CTX_STACK)
end

function empty_ctx_stack()
    global CTX_STACK = []
end

function get_names(kwargs)
    names = get(kwargs, :names, nothing)
    names == nothing ?
        (get(kwargs, :prefix, false) != false && get(kwargs, :num, false) != false ?
         [(kwargs[:prefix], i) for i in range(1, length = kwargs[:num])] : nothing) :
         convert(Vector{Any}, names)
end

function def_species(; type = Continuous, kwargs...)
    kwargs = Dict(kwargs)
    ns = get(kwargs, :ns, (:default, type))
    species = get_names(kwargs)
    @assert(species != nothing)
    if get(CUR_CTX[:species], ns, false) == false
        CUR_CTX[:species][ns] = Dict(:x => species, :type => type, :prior => get(kwargs, :prior, 2^31-1))
    else
        @assert(CUR_CTX[:species][ns][:type] == type)
        append!(CUR_CTX[:species][ns][:x], species)
    end
end

function def_multi_pools(; kwargs...)
    kwargs = Dict(kwargs)
    ns = get(kwargs, :ns, (:default,))
    names = get_names(kwargs)
    raw = get(kwargs, :raw, nothing)
    type = get(kwargs, :re_type, nothing)
    species = get(kwargs, :species, [mmask((:default,))])
    @assert(names != nothing)
    if get(CUR_CTX[:multi_pools], ns, false) == false
        CUR_CTX[:multi_pools][ns] = Dict(:x => names, :species => species, :raw => raw, :re_type => type)
    else
        append!(CUR_CTX[:multi_pools][ns][:x], names)
    end
end

const DEFAULT_RULE_TYPE = ReactiveRule{MassActionContinuous}

function def_rule(; kwargs...)
    kwargs = Dict(kwargs)
    ns = get(kwargs, :ns, :default)
    @assert(!haskey(CUR_CTX[:rules],ns))
    @assert(haskey(kwargs, :params))
    CUR_CTX[:rules][ns] = Dict(:type => get(kwargs, :type, DEFAULT_RULE_TYPE),
                               :label => ns,
                               :params => kwargs[:params],
                               :species => get(kwargs, :species, mmask((:default,))))
end

function get_weights(kwargs)
    haskey(kwargs, :weights) ?
        kwargs[:weights] :
        [(rep_any,) => ((rep_any,) => 1.)] # multi_pool_mask -> species_mask -> weight
end

function apply_rule(; kwargs...)
    kwargs = Dict(kwargs)
    ns = get(kwargs, :ns, :default)
    weights = get_weights(kwargs)
    @assert(haskey(kwargs, :states))
    push!(CUR_CTX[:rules_applications], Dict(:ns => ns,
                                             :states => kwargs[:states],
                                             :alloc_type => get(kwargs, :alloc_type, :alloc_all),
                                             :weights => weights))
end

function apply_rules(; kwargs...)
    kwargs = Dict(kwargs)
    weights = get_weights(kwargs)
    @assert(haskey(kwargs, :states))
    @assert(haskey(kwargs, :rules))
    foreach(kwargs[:rules]) do rule_ns
        apply_rule(ns = rule_ns, states = kwargs[:states],
                   alloc_type = get(kwargs, :alloc_type, :alloc_all),
                   weights = weights)
    end
end

function def_state(; kwargs...)
    kwargs = Dict(kwargs)
    ns = get(kwargs, :ns, :default)
    weights = haskey(kwargs, :prod_weights) ?
        kwargs[:prod_weights] :
        [(rep_any,) => ((rep_any,) => 1.)] # multi_pool_mask -> species_mask -> weight
    do_transition = !get(kwargs, :no_transition, false)
    @assert(!haskey(CUR_CTX[:states], ns))
    CUR_CTX[:states][ns] = Dict(
        :alloc_prod_multi_pools => kwargs[:alloc_prod_multi_pools],
        :trans_multi_pools_in => do_transition ? get(kwargs, :trans_multi_pools_in, kwargs[:alloc_prod_multi_pools]) : [],
        :trans_multi_pools_out => do_transition ? get(kwargs, :trans_multi_pools_out, kwargs[:alloc_prod_multi_pools]) : [],
        :prod_weights => weights,
        :prior => get(kwargs, :prior, 2^31),
        :rules => [])
end

function def_states(; kwargs...)
    kwargs = Dict(kwargs)
    names = get_names(kwargs)
    foreach(ns -> def_state(ns = ns, alloc_prod_multi_pools = [ns]), names)
end

function def_trans_mtx(;kwargs...)
    kwargs = Dict(kwargs)
    ns = get(kwargs, :ns, :default)
    @assert(!haskey(CUR_CTX[:trans_mtxs], ns))
    CUR_CTX[:trans_mtxs][ns] = Dict(
        :trans_mtx => kwargs[:trans_mtx]
    )
end

function def_transition(;kwargs...)
    kwargs = Dict(kwargs)
    ns = get(kwargs, :ns, :default)
    @assert(!haskey(CUR_CTX[:transitions], ns))
    CUR_CTX[:transitions][ns] = Dict(
        :type => get(kwargs, :type, :constant),
        :args => kwargs[:args]
    )
end

function def_system(; kwargs...)
    kwargs = Dict(kwargs)
    ns = get(kwargs, :ns, :default)
    @assert(!haskey(CUR_CTX[:systems], ns))
    CUR_CTX[:systems][ns] = Dict(
        :species => get(kwargs, :species, [mitem(any_item_p)]),
        :multi_pools => get(kwargs, :multi_pools, [mitem(any_item_p)]),
        :rules => get(kwargs, :rules, [mitem(any_item_p)]),
        :states => get(kwargs, :states, (rep_any,)),
        :transition => get(kwargs, :transition, :default),
        :init => get(kwargs, :init, true)
    )
end


####

get_in(x, ks) = reduce((xx, k) -> xx == nothing ? nothing : get(xx, k, nothing), ks, init = x)

function match_path(pvs, x, path, mask)
    try
        ks = keys(x)
        # println("match_path $path, $ks, $(match_mask(mask, path))")
        foreach(ks) do k
            p = (path..., k)
            if match_mask(mask, p)
                push!(pvs, (p, get(x, k, [])))
            end
            match_path(pvs, get(x, k, []), p, mask)
        end
    catch _
    end

    pvs
end

function get_in_mask(x, ks_mask)
    cmask = compile_mask(ks_mask)
    pvs = match_mask(cmask, ()) ? [x] : []
    # map(pv -> pv[2], match_path(pvs, x, (), cmask))
    match_path(pvs, x, (), cmask)
end

#=

=#

function make_index(col)
    reduce(col |> enumerate, init = Dict()) do index, (i, c)
        index[c] = i; index
    end
end

# function make_index(col)
#     index = Dict()
#
#     foreach(col |> enumerate) do (i, c)
#         index[c] = i
#     end
#
#     index
# end

sort_pools(pools) = sort(values(pools) |> collect, by = sp -> sp[1][2][:prior])

function canonize_species(ctx, species_msk = [mitem(any_item_p)])
    species_info = reduce(unique(species_msk), init = []) do sp, msk
        append!(sp, get_in_mask(ctx, (beg, :species, msk,  term)))
    end

    pools = Lazy.groupby(s -> s[2][:type], sort(species_info, by = s -> s[2][:prior]))

    species = reduce(sort_pools(pools), init = []) do sp, ps
        reduce((s, p) -> append!(s, p[2][:x]), ps, init = sp)
    end

    species_by_pool = reduce(sort_pools(pools) |> enumerate, init = Dict()) do sp, (pidx, ps)
        ss = reduce((s, p) -> append!(s, p[2][:x]), ps, init = [])
        reduce(ss |> enumerate, init = sp) do sp, (i, s)
            sp[s] = (pidx, i); sp
        end
    end

    Dict(:species => species,
         :species_by_pool_idxs => species_by_pool,
         :pools => pools,
         :species_idxs => make_index(species))
end

function species_mapping(;ctx = nothing, species_msk = [mitem(any_item_p)])
    ctx = ctx == nothing ? CUR_CTX : ctx

    canonize_species(ctx, species_msk)[:species_by_pool_idxs]
end

pool_size(info) = @>> info map(x->length(x[2][:x])) sum

pools_sizes(pools_info) = map(pool_size, sort_pools(pools_info))

function make_multi_pool(species_info, name, raw, re_type)
    if raw == nothing
        si = sort(species_info[:pools] |> collect, by = s -> s[2][1][2][:prior])
        pools = reduce(si, init = Any[]) do pools, (tp, info)
            l = pool_size(info)
            # println("make_multi_pool: $tp,  $l ", typeof(deepfill(neutral_el(tp), l)))
            tp = re_type == nothing ? tp : get(re_type, (name, tp), tp)
            push!(pools, Pool{tp,1}(deepfill(neutral_el(tp), l), (name, tp)))
        end

        ArrayPartition(pools...)
    else
        raw
    end
end

function mk_pools_info(ctx, pool_msk)
    reduce(unique(pool_msk), init = []) do pools, msk
        append!(pools, get_in_mask(ctx, (beg, :multi_pools, msk, term)))
    end
end

function mk_mpools(pools_info)
    reduce(pools_info, init = []) do mps, pi
        append!(mps, pi[2][:x])
    end
end

function make_multi_pools(ctx, pool_msk, species_info)
    pools_info = mk_pools_info(ctx, pool_msk)

    mpools = mk_mpools(pools_info)
    mpools_raw = reduce(pools_info, init = []) do mps, pi
        append!(mps, pi[2][:raw] != nothing ? pi[2][:raw] : fill(nothing, length(pi[2][:x])))
    end
    mpools_re_type = reduce(pools_info, init = []) do mps, pi
        append!(mps, fill(pi[2][:re_type], length(pi[2][:x])))
    end

    mps = map(partial(make_multi_pool, species_info), mpools, mpools_raw, mpools_re_type)

    Dict(:multi_pools => length(mps) < 500 ? ArrayPartition(mps...) : BigArrayPartition(mps),
         :multi_pool_idx => make_index(mpools),
         :info => pools_info)
end

function multi_pools_mapping(;ctx = nothing, pool_msk = [mitem(any_item_p)])
    ctx = ctx == nothing ? CUR_CTX : ctx

    mk_pools_info(ctx, pool_msk) |> mk_mpools |> make_index
end

function def_multi_pools_with_species(def :: Dict)
    @assert haskey(def, :species)
    @assert haskey(def, :multi_pools)

    foreach(def[:species] |> enumerate) do (i, s)
        @assert haskey(s, :names)
        def_species(type = get(s, :type, Continuous), names = s[:names], prior = i)
    end

    foreach(def[:multi_pools] |> enumerate) do (i, mp)
        @assert haskey(mp, :names)
        def_multi_pools(names = mp[:names], re_type = get(mp, :re_type, nothing))
    end
end

function make_multi_pools_species_maps(def :: Dict)
    push_ctx()
    try
        def_multi_pools_with_species(def)

        spec_map = species_mapping()
        mp_map = multi_pools_mapping()
        mp_spec_map = reduce(mp_map |> keys |> collect, init = Dict()) do sp_mp, mp
            reduce(spec_map |> keys |> collect, init = sp_mp) do sp_mp, sp
                sp_mp[(mp, sp)] = (mp_map[mp], spec_map[sp]...); sp_mp
            end
        end

        Dict(
            :spec => spec_map,
            :mp => mp_map,
            :mp_spec => mp_spec_map
        )
    finally
        pop_ctx()
    end
end

#=

=#

array_part(x) = ArrayPartition(x...)

part_by_spec_type(rs, ps_size, ps) = @>> partition_mtx_by_cols(reshape(rs, (1, ps)), ps_size) map(x -> reshape(x, length(x)))

function expand_reqs(reqs, ps_size, species_info)
    reqs = partition(2, reqs)
    ps = sum(ps_size)
    rs = convert(Array{Any}, fill(0., ps))
    sps = species_info[:species_idxs]
    foreach(reqs) do (n, s)
        rs[sps[s]] = n
    end
    part_by_spec_type(rs, ps_size, ps) |> array_part
end

function react_rule_parse(req, species_info)
    ps_size = pools_sizes(species_info[:pools])
    req, prod = splitby(x -> !(typeof(x) <: Tuple && typeof(x[1]) <: Symbol), req)
    (_, rk) = first(prod)
    prod = tail(prod)
    expand_reqs(req, ps_size, species_info), expand_reqs(prod, ps_size, species_info), rk
end

function make_rule(r_type :: Type{T}, name, params, species_info) where T <: GeneralReactiveRule
    req, prod, rk = react_rule_parse(params, species_info)
    r_type(req, prod, rk, name)
end

function make_rule(r_type :: Type{ReactiveRuleGroup}, name, params, species_info)
    ReactiveRuleGroup(AbstractRule[NopRule(p) for p in params], name)
end

function make_rule(r_type :: Type{TimeDelayedRule}, name, params, species_info)
    TimeDelayedRule(NopRule(params[1]), params[2], name)
end

function update_rule(rule :: T, rules, rule_idx) where T <: GeneralReactiveRule end

function update_rule(rule :: ReactiveRuleGroup, rules, rule_idx)
    foreach(enumerate(rule.rules)) do (i, r)
        rule.rules[i] = rules[rule_idx[r.label]]
    end
end

function update_rule(rule :: TimeDelayedRule, rules, rule_idx)
    rule.rule = rules[rule_idx[rule.rule.label]]
end

function make_rules(ctx, rules_mask, species_info)
    rules_info = reduce(unique(rules_mask), init = []) do rules, msk
        append!(rules, get_in_mask(ctx, (beg, :rules, msk,  term)))
    end

    rules = map(rules_info) do (_, info)
        make_rule(info[:type], info[:label], info[:params], species_info)
    end

    rule_idx = Dict()
    foreach(enumerate(rules)) do (i, r)
        rule_idx[r.label] = i
    end

    foreach(r -> update_rule(r, rules, rule_idx), rules)

    Dict(:rules => rules,
         :rule_idx => rule_idx,
         :info => rules_info)
end

#=

=#

function make_weights(multi_pools, multi_pools_info, species_info, weight_masks)
    ps_size = pools_sizes(species_info[:pools])
    ps = sum(ps_size)

    mps_idx = Dict()
    for (i, mp) in enumerate(multi_pools)
        mps_idx[mp] = i
    end

    ws = map(_ -> fill(0, ps), multi_pools)

    foreach(weight_masks) do weight_mask
        (mp_mask, (spec_mask, w)) = weight_mask
        mps = filter(mp -> match_mask(mp_mask, mp), multi_pools)
        sps = filter(sp -> match_mask(spec_mask, sp), species_info[:species])

        for mp in mps
            for sp in sps
                ws[mps_idx[mp]][species_info[:species_idxs][sp]] = w
            end
        end
    end

    ws
end

function apply_rules_on_states(ctx)
    states_name = ctx[:states] |> keys

    foreach(ctx[:rules_applications]) do ra
        sts = reduce(ra[:states], init = []) do sts, s_msk
            append!(sts, filter(s -> match_mask(s_msk, s), states_name))
        end

        foreach(sts) do s
            push!(ctx[:states][s][:rules], ra)
        end
    end
end

#=
w = make_weights([(:places,1), (:places, 2)], mps_info, s_info, [(:places,1) => ((rep_any,) => 1.), (:places,2) => ((rep_any,)=>2.)])

(mp_msk, (sp_mask, w)) = rep_any => (rep_any => 1.)
match_mask((:places,1), (:places,1))
CUR_CTX[:states][(:s,1)]
CUR_CTX[:rules_applications]
=#

function state_multipools_idxs(multi_pools_info, mps)
    map(mp -> multi_pools_info[:multi_pool_idx][mp], mps)
end

function state_multipools(multi_pools_info, mp_masks)
    mp_names = keys(multi_pools_info[:multi_pool_idx])
    mps = reduce(mp_masks, init = []) do mps, msk
        append!(mps, filter(mp -> match_mask(msk, mp), mp_names))
    end |> unique

    Dict(:names => mps,
         :idxs => state_multipools_idxs(multi_pools_info, mps))
end

function reshape_to_mpool_rule_spec(rules_weights)
    if isempty(rules_weights)
        []
    else
        reduce(rules_weights, init = [[] for _ in first(rules_weights)]) do rws, rw
            foreach((ws, w) -> push!(ws, w), rws, rw)
            rws
        end
    end
end

function make_state(rules_info, multi_pools_info, species_info, state)
    ap_mpools = state_multipools(multi_pools_info, state[:alloc_prod_multi_pools])
    rules = map(r -> rules_info[:rule_idx][r[:ns]], state[:rules])
    rules_alloc = map(r -> r[:alloc_type], state[:rules])
    rules_weights = map(r -> make_weights(ap_mpools[:names], multi_pools_info, species_info, r[:weights]),
                        state[:rules]) |> reshape_to_mpool_rule_spec
    ps_size = pools_sizes(species_info[:pools])
    ps = sum(ps_size)
    weights = make_weights(ap_mpools[:names], multi_pools_info, species_info, state[:prod_weights])
    prod_weights = map(weights) do ws
        part_by_spec_type(ws, ps_size, ps)
    end
    trans_in = state_multipools(multi_pools_info, state[:trans_multi_pools_in])
    trans_out = state_multipools(multi_pools_info, state[:trans_multi_pools_out])

    StateDesc(ap_mpools[:idxs],
              rules,
              rules_alloc,
              rules_weights,
              prod_weights,
              trans_in[:idxs],
              trans_out[:idxs])
end

function make_states(ctx, state_mask, rules_info, multi_pools_info, species_info)
    names = filter(s -> match_mask(state_mask, s), keys(ctx[:states])) |> collect
    sts = @> map(s -> ctx[:states][s], names) sort(by = x -> x[:prior])

    states = map(sts) do state
        make_state(rules_info, multi_pools_info, species_info, state)
    end

    Dict(
        :states => states,
        :state_idx => make_index(names)
    )
end

function make_trans_mtx_(state_info, trans_mtx_info)
    l = state_info[:states] |> length
    trans_mtx = fill(0., (l, l))

    foreach(trans_mtx_info[:trans_mtx]) do (from, (to, w))
        trans_mtx[state_info[:state_idx][to], state_info[:state_idx][from]] = w
    end

    s = sum(trans_mtx, dims = 1)
    trans_mtx .= trans_mtx ./ (s |> epsilonize)
    # trans_mtx[diagind(trans_mtx)] .= (x -> x == 0 ? 1 : x).(reshape(s, l))
    for i in 1:l
        if s[i] == 0
            trans_mtx[i,i] = 1.
        end
    end

    trans_mtx
end

function make_trans_mtxs(ctx, state_info)
    trans_mtxs = map(ctx[:trans_mtxs] |> values) do tm
        make_trans_mtx_(state_info, tm)
    end

    Dict(
        :trans_mtxs => trans_mtxs,
        :trans_mtx_idx => make_index(@> ctx[:trans_mtxs] keys)
    )
end

#=
=#

function make_transition_(trans_fn, trans_info, trans_mtxs_info, species_info)
    trans_mtxs = Any[nothing for _ in species_info[:species]]

    foreach(trans_info[:args]) do (msk, mtx)
        specs = filter(s -> match_mask(msk, s), species_info[:species])
        foreach(specs) do s
            idx = trans_mtxs_info[:trans_mtx_idx][mtx]
            trans_mtxs[species_info[:species_idxs][s]] = trans_mtxs_info[:trans_mtxs][idx]
        end
    end

    trans_fn([trans_mtxs |> make_trans_mtx_compact])
end


function make_transition(::Type{Val{:constant}}, trans_info, trans_mtxs_info, species_info)
    make_transition_(const_trans, trans_info, trans_mtxs_info, species_info)
end

function make_transition(::Type{Val{:ts_constant}}, trans_info, trans_mtxs_info, species_info)
    make_transition_(ts_const_trans, trans_info, trans_mtxs_info, species_info)
end

function make_transition(trans_info, trans_mtxs_info, species_info)
    make_transition(Val{trans_info[:type]}, trans_info, trans_mtxs_info, species_info)
end

function make_system(ctx, system_ns)
    system_info = ctx[:systems][system_ns]

    s_info = canonize_species(ctx, system_info[:species])
    mps_info = make_multi_pools(ctx, system_info[:multi_pools], s_info)
    rs_info = make_rules(ctx, system_info[:rules], s_info)
    sts_info = make_states(ctx, system_info[:states], rs_info, mps_info, s_info)
    tmtxs_info = make_trans_mtxs(ctx, sts_info)
    transition = haskey(ctx[:transitions], system_info[:transition]) ?
        make_transition(ctx[:transitions][system_info[:transition]], tmtxs_info, s_info) :
        (_,_) -> nothing

    species = map(s_info[:species] |> enumerate) do s
        Species(s...)
    end

    system = System(transition, mps_info[:multi_pools], rs_info[:rules], species, sts_info[:states], Dict())
    if system_info[:init]
        init_system!(system)
    end
    Dict(
        :system => system,
        :species_info => s_info,
        :multi_pools_info => mps_info,
        :rules_info => rs_info,
        :states_info => sts_info,
        :transition_matrices_infp => tmtxs_info,
        :transition => transition
    )
end

function make_systems(ctx)
    apply_rules_on_states(CUR_CTX)

    systems_info = Dict()
    foreach(ctx[:systems] |> keys) do sns
        systems_info[sns] = make_system(ctx, sns)
    end

    systems_info
end

@inline function get_system_info(systems_info, ns)
    systems_info[ns]
end

@inline function get_system(systems_info, ns)
    systems_info[ns][:system]
end

function setup_multi_pools(system_info, mutli_pool_setup)
    multi_pools_info = system_info[:multi_pools_info]
    multi_pools_names = multi_pools_info[:multi_pool_idx] |> keys
    multi_pools = multi_pools_info[:multi_pools]
    species_info = system_info[:species_info]

    foreach(mutli_pool_setup) do mp_setup
        (mp_mask, (spec_mask, v)) = mp_setup
        mps = filter(mp -> match_mask(mp_mask, mp), multi_pools_names)
        sps = filter(sp -> match_mask(spec_mask, sp), species_info[:species])

        for mp in mps
            for sp in sps
                multi_pools.x[multi_pools_info[:multi_pool_idx][mp]][species_info[:species_idxs][sp]] = v
            end
        end
    end
end

function new_default_steady_ctx()
    new_ctx()
    def_trans_mtx(trans_mtx = [])
    def_transition(args = [(rep_any,) => :default])
    def_system()
end

#=
new_ctx()
def_species(type = Continuous, names = Any[:s, :e, :i, :r])
def_species(prefix = :a, num = 10)
def_species(type = Discrete, prefix = :c, num = 10)
def_species(ns = :aa, prefix = :b, num = 10)

def_multi_pools(prefix = :places, num = 10)

def_rule(ns = :si_e, params = [1,:s,1,:i,(:->, 0.9),1,:e])
def_rule(ns = :e_i, params = [1,:e,(:->, 0.3), 1, :i])
def_rule(ns = :i_r, params = [1,:i,(:->, 0.03), 1, :r])
def_rule(ns = :r_s, params = [1,:r,(:->, 0.003), 1, :s])
def_rule(type = ReactiveRuleGroup, ns = :seir, params = [:si_e, :e_i, :i_r, :r_s])

def_state(ns = (:s,1), alloc_prod_multi_pools = [(:places, 1)])

apply_rule(ns = :seir, states = [(rep_any,)])

def_trans_mtx(trans_mtx = [])
def_transition(args = [(rep_any,) => :default])
def_system()

apply_rules_on_states(CUR_CTX)
system_info = make_system(CUR_CTX, :default)

#
s_info
make_trans_mtx(sts_info, CUR_CTX[:trans_mtxs][:default])

tr = make_transition(Val{:constant}, CUR_CTX[:transitions][:default], tmtxs_info, s_info)

s_info = canonize_species(CUR_CTX, [mmask((:default,)), :aa])
mps_info = make_multi_pools(CUR_CTX, [(:default,)], s_info)
rs_info = make_rules(CUR_CTX, [mitem(any_item_p)] , s_info)

sts_info = make_states(CUR_CTX, (rep_any,), rs_info, mps_info, s_info)
tmtxs_info = make_trans_mtxs(CUR_CTX, sts_info)
trans_info = make_transitions(CUR_CTX, tmtxs_info, s_info)


=#

#=

plicni fibroza, gallogy

=#
