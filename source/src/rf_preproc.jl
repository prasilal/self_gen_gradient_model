## Processing tools

filtered_copy(pred, x) = @>> x copy filter(pred)
filtered_copy!(pred, x) = @>> x copy filter!(pred)

@inline function foreach_item(f, m)
    finished = false
    @inbounds for i in 1:length(m)
        @inbounds for j in 1:length(m[i])
            finished = f(i,j)
            if finished == true
                break
            end
        end
        if finished == true
            break
        end
    end
end

function expand_rules(rules :: Vector{<:AbstractRule}, rs, state :: StateDesc, getter :: Function)
    reduce((rs, r) ->
             map((cs, c) -> push!(cs, c),
                 rs, getter(rules[r], PARTITION)),
           state.rules;
           init = rs)
end

vstack(x) = hcat(x...) |> transpose

function make_compact_stochiometry(rules  :: Vector{<:AbstractRule}, states :: Vector{StateDesc}, getter :: Function)
    reduce((rs, s) -> expand_rules(rules, rs, s, getter),
                  states;
                  init = [[] for _ in getter(rules[1], PARTITION)])
end

function prepare_requirements(rules :: Vector{<:AbstractRule}, states :: Vector{StateDesc})
    reqs = make_compact_stochiometry(rules, states, get_reactants_stochiometry)
    map(vstack, reqs)
end

function partition_mtx_by_cols(m, spans)
    reduce(
        ((ms, shift), l) ->
          (push!(ms, m[:, shift:(shift + l - 1)]), shift + l),
        spans;
        init = ([], 1)
    ) |> first
end

function zero_weights(rules_weights)
    map(ws -> zeros(length(ws)), rules_weights)
end

function add_state_weights(rs, s, mpi)
    idx = findfirst(x -> x == mpi, s.multi_pools)
    ws = idx == nothing ?
        zero_weights(s.rules_weights[1]) :
        s.rules_weights[idx]
    append!(rs, ws)
end

function prepare_rules_weights_for_mpool(states :: Vector{StateDesc}, reqs, mpi)
    weights = reduce((rs, s) -> add_state_weights(rs, s, mpi), states;
                     init = []) |> vstack
    partition_mtx_by_cols(weights,
                          map(r -> r[1,:] |> length, reqs))
end

function prepare_rules_weights(states :: Vector{StateDesc}, reqs, mpools)
    map(mpi -> prepare_rules_weights_for_mpool(states, reqs, mpi),
        range(1, length=length(mpools.x)))
end

function make_rules_by_state(states, rules)
    reduce((rs, s) -> append!(rs, map(r -> (r,s), rules[s.rules])), states; init = [])
end

function prepare_rules_mask(states :: Vector{StateDesc}, mpools)
    states_rule_pools = reduce((rs, s) -> append!(rs, map(_ -> s.multi_pools, s.rules) |> collect), states, init = [])
    mpool_mask = zeros(length(states_rule_pools), length(mpools.x))
    for (idx, mps) in enumerate(states_rule_pools)
        mpool_mask[idx, mps] .= 1
    end
    mpool_mask
end

function expand_masks(msks, state, i)
    ws = i in state.multi_pools ?
        state.species_sinks_weights[findfirst(x -> x == i, state.multi_pools)] :
        [zeros(length(i)) for i in state.species_sinks_weights[1]]
    reduce((masks, _) -> map((m, w) -> push!(m, w), masks, ws),
           state.rules; init = msks)
end

function make_production_masks(mpools, states :: Vector{StateDesc})
    prod_masks = []
    for i in range(1, length=length(mpools.x))
        masks = reduce((msks, state) -> expand_masks(msks, state, i),
                       states;
                       init = [[] for _ in states[1].species_sinks_weights[1]])
        push!(prod_masks, map(vstack, masks))
    end
    prod_masks
end

function make_rows_allocators(states :: Vector{StateDesc})
    map(x -> Val{x}(), reduce((ra, s) -> vcat(ra, s.rules_allocators), states, init = []))
end

Base.eps(x :: Type{T}) where {T <: Integer} = 1

@inline epsilonize(m) = map.(x -> ifelse(x == 0, eps(typeof(x)), x), m)

function reqs_indicator(reqs)
    map(r -> r .!= 0, reqs)
end

function transition_mpools(states)
    reduce((mps, s) -> union!(mps, s.multi_pools_transition_in, s.multi_pools_transition_out),
           states, init = Set{Int}())
end

function del_agents!(a, els)
    foreach(el -> filter!(e -> e.uid != el.uid, a), els)
end

liftx(mp :: AbstractPool{T,N}) where {T, N} = mp.x
liftx(mp :: ArrayPartition) = mp.x
liftx(mp :: BigArrayPartition) = mp.x
liftx(mp) = mp
liftxs(mps) = map(liftx, mps)

function foreach_liftxs(f, args...)
    foreach((as...) -> f(as...),
            liftxs(args)...)
end

const foreach_pool = foreach_liftxs

function foreach_pool_item(f, args...)
    foreach_pool((as...) -> foreach((bs...) -> f(bs...), as...),
                 args...)
end
