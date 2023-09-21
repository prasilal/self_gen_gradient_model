function marray2aarray(m :: AbstractArray{T,N}; dims = 1) where {T, N}
    idx = [i == dims ? 0 : (:) for i in range(1, length = N)]
    reduce((arr, i) -> (idx[dims] = i; push!(arr, m[idx...])),
           range(1, length = size(m)[dims]),
           init = Vector{Array{T, N-1}}())
end

# TODO: use continuous multinomial dist
function prepare_mpool_distribution(pool :: AbstractArray{T, N}, probs) where {T <: AbstractFloat, N}
    # println("prepare_mpool_distribution", pool, probs, marray2aarray(probs, dims = 2))
    map(p -> pool .* p, marray2aarray(probs, dims = 2))
end

# function prepare_mpool_distribution(pool :: AbstractArray{T, N}, probs) where {T <: Integer, N}
#     marray2aarray(hcat(rand.((x->Multinomial(x, probs)).(pool))...))
# end

function discrete_dist(pool :: AbstractArray{T, N}, probs) where {T <: Integer, N}
    ps = marray2aarray(probs)
    map((s, ps) -> rand(Multinomial(s, ps)), pool, ps) |> hcat |> marray2aarray
end

function prepare_mpool_distribution(pool :: AbstractArray{T, N}, probs) where {T <: Integer, N}
    discrete_dist(pool, probs)
end

function prepare_mpool_distribution(pool :: AbstractArray{T, N}, probs) where {T <: Condition, N}
    discrete_dist(length.(pool), probs)
end

function distribute_agents(agents, dist)
    tmp_as = copy(agents)
    map(dist -> splice!(tmp_as, 1:dist), dist)
end

function per_item_dist(per_item_fn, pool, probs)
    ps = marray2aarray(probs)
    das = map(per_item_fn, pool, ps)
    # println("prepare_mpool_distribution: $pool $ps $das")
    map((xs...) -> [xs...], das...)
end

function prepare_mpool_distribution(pool :: AbstractArray{T, N}, probs) where {T <: Agents, N}
    per_item_dist((s, ps) -> distribute_agents(s, rand(Multinomial(length(s), ps))), pool, probs)
end

function prepare_mpool_distribution(pool :: AbstractArray{T, N}, probs) where {T <: AbstractArray{TT ,NN} where {TT <: AbstractFloat, NN}, N}
    per_item_dist((s, ps) -> map(p -> s .* p, ps), pool, probs)
end

function update_pool_trans!(dest_pool, src_pool :: AbstractArray{T, N}) where {T <: Real, N}
    dest_pool .+= src_pool
end

function update_pool_trans!(dest_pool, src_pool :: AbstractArray{T, N}) where {T <: Condition, N}
    foreach((d, s) -> append!(d,s), dest_pool, src_pool)
end

function update_pool_trans!(dest_pool, src_pool :: AbstractArray{T, N}) where {T <: Agents, N}
    foreach((d, s) -> append!(d,s), dest_pool, src_pool)
end

function update_pool_trans!(dest_pool, src_pool :: AbstractArray{T, N}) where {T <: AbstractArray{TT ,NN} where {TT <: AbstractFloat, NN}, N}
    foreach((d, s) -> d .+= s, dest_pool, src_pool)
end

function prep_state_trans(mpools, state, states, probs, src_state; pools_mask = nothing)
    prep_mpools = Dict{Int, Any}()
    foreach(state.multi_pools_transition_out) do mpi
        prep_mpools[mpi] = map(mpools.x[mpi].x, probs, pools_mask) do pool, p, msk
            msk && prepare_mpool_distribution(pool, p[:, :, src_state])
        end
    end
    prep_mpools
end

function comp_state_transition!(mpools_update, mpools, state, states, probs, src_state; pools_mask = nothing)
    prep_mpools = prep_state_trans(mpools, state, states, probs, src_state, pools_mask = pools_mask)
    for (state_idx, dest_state) in enumerate(states)
        l = length(dest_state.multi_pools_transition_in)
        np = 1. / l
        # local_probs = fill(np, l)
        for mpi in state.multi_pools_transition_in
            # println("comp_state_trans: $src_state $state_idx $mpi $(prep_mpools[mpi])")
            prep = map(prep_mpools[mpi], pools_mask) do p, msk
                msk && prepare_mpool_distribution(p[state_idx],
                                                  fill(np, (length(p[state_idx]), l)))
            end
            for (i, dmpi) in enumerate(dest_state.multi_pools_transition_in)
                foreach(mpools_update[dmpi], prep, pools_mask) do d, s, msk
                    msk && update_pool_trans!(d, s[i])
                end
            end
        end
    end
end

function update_mpool!(dest_pool :: Pool{T, N}, src_pool) where {T <: Real, N}
    dest_pool.x = src_pool
end

function update_mpool!(dest_pool :: Pool{T, N}, src_pool) where {T <: Condition, N}
end

function update_mpool!(dest_pool :: Pool{T, N}, src_pool) where {T <: Agents, N}
    if length(dest_pool.x) > 0
        tp = eltype(typeof(dest_pool.x[1])) # pools should have at least one specie
        tmp = reduce((p, a) -> push!(p, ifelse(isempty(a), Agent{tp}[], a)), src_pool, init = Agents{tp}[])
        dest_pool.x = tmp
    end
end

function update_mpool!(dest_pool :: Pool{T, N}, src_pool) where {T <: AbstractArray{TT ,NN} where {TT <: AbstractFloat, NN}, N}
    for i in range(1, length = dest_pool.x |> length)
        dest_pool.x[i] = src_pool[i]
    end
end

function transition_update!(mpools, trans_mpools, mpools_update; pools_mask = nothing)
    for (i, mpool) in enumerate(mpools.x)
        if i in trans_mpools
            foreach((dp, sp, msk) -> msk && update_mpool!(dp, sp), mpool.x, mpools_update[i], pools_mask)
        end
    end
end

function transition!(mpools, trans_mtx, states, trans_mpools; pools_mask = nothing)
    pools_mask = pools_mask == nothing ? fill(true, length(mpools.x[1].x)) : pools_mask

    mpools_update = [empty_allocation(mp, mp.x) for mp in mpools.x]
    for (i, state) in enumerate(states)
        comp_state_transition!(mpools_update, mpools, state, states, trans_mtx, i, pools_mask = pools_mask)
    end

    transition_update!(mpools, trans_mpools, mpools_update, pools_mask = pools_mask)
end

df(t, pool) = deepfill(neutral_el(t), size(pool))

# TODO: not defined for now
function filtered_mpool(pred :: Function, pool :: AbstractArray{T, N}) where {T <: Real, N}
    df(T, pool)
end

# TODO: not defined for now
function filtered_mpool(pred :: Function, pool :: AbstractArray{T, N}) where {T <: Condition, N}
    df(T, pool)
end

function filtered_mpool(pred :: Function, pool :: AbstractArray{T, N}) where {T <: Agents, N}
    map(partial(filtered_copy, pred), pool)
end

function filtered_mpool(pred :: Function, mpool)
    map(partial(filtered_mpool, pred), mpool.x)
end

# TODO: not defined for now
function remove_from!(dest :: AbstractArray{T,N}, src) where {T <: Real, N}
    dest
end

# TODO: not defined for now
function remove_from!(dest :: AbstractArray{T,N}, src) where {T <: Condition, N}
    dest
end

function remove_from!(dest :: AbstractArray{T,N}, src) where {T <: Agents, N}
    foreach((d, s) -> del_agents!(d, s), dest, src)
end

function update_filtered_mpools(mpools_tmp, states, trans_mpools, cond_t)
    pred, trans_mtx = cond_t
    f_mpools = pred <: Function ? filtered_mpool(pred, mpools_tmp) : mpools_tmp
    foreach_pool((dp, sp) -> remove_from!(dp, sp), mpools_tmp, f_mpools)
    transition!(f_mpools, trans_mtx, states, trans_mpools)
    foreach_pool((dp, sp) -> update_pool_trans!(dp, sp), mpools_tmp)
end

function conditional_transition!(mpools, states, trans_mpools, conds_ts)
    mpools_tmp = deepcopy(mpools)
    mpools_update = [empty_allocation(mp, mp.x) for mp in mpools.x]

    foreach(partial(update_filtered_mpools, mpools_tmp, states, trans_mpools),
            conds_ts)

    transition_update!(mpools, trans_mpools, mpools_update)
end
