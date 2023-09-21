function comp_coeffs(reqs :: Req{T, N, Val{:alloc_all}}, prior_weights) where {T, N}
    reqs.x
end

function comp_coeffs(reqs :: Req{T, N, AC}, prior_weights) where {T, N, AC}
    coeffs = reqs.x .* prior_weights
    totals = sum(coeffs, dims = 1)
    coeffs ./ epsilonize(totals)
end

# continuous prealloc
function prealloc(pool :: Pool{T,N}, reqs, weights, _) where {T <: AbstractFloat, N}
    # println("prealloc ", reqs, weights)
    coeffs = comp_coeffs(reqs, weights)
    # println("prealloc reqs: $reqs ws: $weights cfs: $coeffs res: $(coeffs .* get_data(pool)')")
    coeffs .* get_data(pool)'
end

# discrete prealloc
function prealloc(pool :: Pool{T,N}, reqs, weights, _) where {T <: Integer, N}
    coeffs = comp_coeffs(reqs, weights)
    floor.(coeffs .* get_data(pool)')
end

function prealloc(pool :: Pool{T,N}, reqs, weights, system) where {T <: Condition, N}
    tmp = map(cond -> cond(system), pool)
    system.cache[:cond_cache][pool.label] = tmp
    @. reqs.x * tmp * weights
end

function prealloc(pool :: Pool{T,N}, reqs, weights, _) where {T <: Agents, N}
    coeffs = comp_coeffs(reqs, weights)
    floor.(coeffs .* length.(get_data(pool))')
end

function prealloc(pool :: Pool{Array{T,NN},N}, reqs :: Req{Tr, NNN, Val{:alloc_all}},
                  weights, _) where {T <: AbstractFloat, N, NN, NNN, Tr}
    coeffs = comp_coeffs(reqs, weights)
    coeffs .* fill(1., get_data(pool) |> length)'
end

function prealloc(pool :: Pool{Array{T,NN},N}, reqs, weights, _) where {T <: AbstractFloat, N, NN}
    coeffs = comp_coeffs(reqs, weights)
    coeffs .* (sum.(get_data(pool)))'
end

# allocated update
function alloc_update!(pool :: Pool{T,N}, _, _, _, reqs, min, min_indices,
                       allocated) where {T <: AbstractFloat, N}
    # println("alloc_update! ", reqs, " ", min, " ", min_indices)
    allocated[min_indices,:] .= reqs[min_indices,:] .* min[min_indices,:]
end

function alloc_update!(pool :: Pool{T,N}, _, _, _, reqs, min, min_indices,
                       allocated) where {T <: Int, N}
    # println("alloc_update! ", reqs, " ", min, " ", min_indices)
    allocated[min_indices,:] .= floor.(reqs[min_indices,:] .* min[min_indices,:])
end

function alloc_update!(pool :: Pool{T,N}, mpools, mpools_ratios, pid, reqs, min, min_indices,
                       allocated) where {T <: Condition, N}
    cc = system.cache[:cond_cache]
    foreach(mp ->
            allocated[min_indices,:] .+= reqs[min_indices,:] .* cc[mp.x[pid].label] .* min[min_indices,:],
            mpools.x)
end

function splice_pool!(pool, rq, row, i, args)
    rng = 1 : convert(Int, reduce((r, m) -> r * m[row,i], args, init = rq) |> floor)
    splice!(pool[i], rng)
end

function update_agent_row!(allocated, pool, row, reqs, args...)
    # println("update_agent_row! ", allocated, " ", pool, " ", row, " ", reqs[row, :], " ", args)
    for (i, r) in enumerate(reqs[row, :])
        if r != 0
            append!(allocated[row,i], splice_pool!(pool, r, row, i, args))
        end
    end
end

function alloc_update!(pool :: Pool{T,N}, mpools, mpools_ratios, pid, reqs, min, min_indices,
                       allocated) where {T <: Agents, N}
    foreach(mpools.x, mpools_ratios) do (mp, r)
        tmp_pool = map(copy, mp.x[pid].x)
        foreach(min_indices) do row
            update_agent_row!(allocated, tmp_pool, row, reqs, min, r[pid])
        end
    end
end

function update_tensor_row!(allocated, pool, row, reqs, args...)
    foreach(reqs[row, :] |> enumerate) do (i, r)
        if r != 0
            allocated[row,i] .= reduce((r, m) -> r * m[row,i], args, init = r) .* pool[i]
        end
    end
end

function alloc_update!(pool :: Pool{Array{T, NN}, N}, mpools, mpools_ratios, pid, reqs, min, min_indices,
                       allocated) where {T <: AbstractFloat, NN, N}
    foreach(mpools.x, mpools_ratios) do (mp, r)
        foreach(min_indices) do row
            update_tensor_row!(allocated, mp.x[pid].x, row, reqs, min, r[pid])
        end
    end
end

function sum_by_row(m)
    s = sum(m, dims = 1)
    as_vector(s)
end

function comp_per_pool_update!(allocated_per_pool, reqs, m, min_indices, ratio, normalizer)
    allocated_per_pool[min_indices,:] .= normalizer.(ratio[min_indices,:] .* reqs[min_indices,:] .* m[min_indices,:])
    sum_by_row(allocated_per_pool[min_indices,:])
end

# continuous allocated pool update
function update_pool!(pool :: Pool{T,N}, reqs, m, min_indices,
                      allocated_per_pool, ratio) where {T <: AbstractFloat, N}
    tmp = comp_per_pool_update!(allocated_per_pool, reqs, m, min_indices, ratio, identity)
    pool.x .-= tmp
end

# discrete allocated pool update
# TODO: update to one with mass preservation
function update_pool!(pool :: Pool{T,N}, reqs, m, min_indices,
                      allocated_per_pool, ratio) where {T <: Integer, N}
    tmp = comp_per_pool_update!(allocated_per_pool, reqs, m, min_indices, ratio, floor)
    pool.x .-= convert(typeof(pool.x), tmp)
end

function update_pool!(pool :: Pool{T,N}, reqs, m, min_indices,
                      allocated_per_pool, ratio) where {T <: Condition, N}
    tmp = comp_per_pool_update!(allocated_per_pool, reqs, system.cache[:cond_cache][pool.label] .* m,
                                min_indices, ratio, identity)
end

function update_pool!(pool :: Pool{T,N}, reqs, m, min_indices,
                      allocated_per_pool, ratio) where {T <: Agents, N}
    tmp_pool = map(copy, pool)
    foreach(row -> update_agent_row!(allocated_per_pool, tmp_pool, row, reqs, m, ratio),
            min_indices)
    foreach((p, t) -> p = t, pool.x, tmp_pool)
end

function update_pool!(pool :: Pool{Array{T, NN},N}, reqs, m, min_indices,
                      allocated_per_pool, ratio) where {T <: AbstractFloat, NN, N}
    foreach(min_indices) do row
        update_tensor_row!(allocated_per_pool, pool, row, reqs, m, ratio)
    end
end

function mpool_update!(mpool_tmp, reqs, mm, min_indices, alloc_per_mpool, ratio)
    foreach((p, rs, per_pool, r, m) -> update_pool!(p, rs, m, min_indices, per_pool, r),
            mpool_tmp.x, reqs, alloc_per_mpool, ratio, mm)
end

function partial_nonzero_mins(pre, reqs)
    l = length(pre)
    a = map((x, y) -> ifelse(y == 0, Inf, x), reshape(pre, l), reshape(reqs, l))
    minimum(reshape(a, size(pre)), dims = 2)
end

# TODO: maybe rename to masked_mins
function nonzero_mins(pre, reqs)
    reduce((ms, r) -> min.(ms, partial_nonzero_mins(r...)), zip(pre, reqs);
                  init = fill(Inf, size(reqs[1])[1]))
end

function comp_prealloc(mpools_tmp, reqs_tmp, weights, mpools_mask, system)
    # println("mpools_tmp ", mpools_tmp)
    # println("reqs_tmp ", reqs_tmp)
    # println("weights ", weights)
    # println("mpools_mask ", mpools_mask)
    # preallocated = empty_allocation(mpools_tmp.x[1], reqs_tmp)
    preallocated = empty_allocation_t(AbstractFloat, reqs_tmp) # is this ok for all types ?
    prealloc_per_mpool = []
    # println("preallocated ", preallocated)
    for (i, mpool_tmp) in enumerate(mpools_tmp.x)
        reqs_tmp1 = map(rs -> Req(rs.x .* mpools_mask[:, i], rs.alloc_type), reqs_tmp)
        # println("reqs_tmp1 ", reqs_tmp1)
        preallocated_part = map((p, reqs, ws) -> prealloc(p, reqs, ws, system),
                                mpool_tmp.x, reqs_tmp1, weights[i])
        # println("preallocated_part ", preallocated_part)
        push!(prealloc_per_mpool, preallocated_part)
        # println("comp_prealloc $preallocated  -- $preallocated_part")
        foreach((p1, p2) -> p1 .+= p2, preallocated, preallocated_part)
    end
    # println("\n")
    prealloc_cannon = epsilonize(preallocated)
    mpools_ratios = map(mp -> map((p, t) -> p ./ t, mp, prealloc_cannon),
                        prealloc_per_mpool)
    preallocated, mpools_ratios
end

function comp_alloc!(mpools_tmp, allocated, allocated_per_mpool, reqs_tmp,
                     mm, min_indices, mpools_ratios)
    foreach((p, pid, reqs, alloc, m) ->
            alloc_update!(p, mpools_tmp, mpools_ratios, pid, reqs, m, min_indices, alloc),
            mpools_tmp.x[1].x, range(1, length=length(mpools_tmp.x)), reqs_tmp,  allocated, mm)

    foreach((mpool_tmp, alloc_per_mpool, r) ->
            mpool_update!(mpool_tmp, reqs_tmp, mm, min_indices,
                          alloc_per_mpool, r),
            mpools_tmp.x, allocated_per_mpool, mpools_ratios)

    allocated, allocated_per_mpool
end

function perform_alloc!(mpools_tmp, reqs_tmp, preallocated, mpools_ratios,
                        allocated, allocated_per_mpool, m, rm, rvals)
    min_indices = findall(x -> x == rm, as_vector(rvals))
    comp_alloc!(mpools_tmp, allocated, allocated_per_mpool,
                reqs_tmp, m, min_indices, mpools_ratios)
    foreach(r -> r[min_indices,:] .= 0., reqs_tmp)
end

function alloc!(::Val{:prealloc_min}, mpools_tmp, reqs_tmp, preallocated, mpools_ratios,
                allocated, allocated_per_mpool, rows_mask)
    # println("alloc!", preallocated, " ", reqs_tmp)
    reqs_tmp1 = map(rq -> rq .* rows_mask, reqs_tmp)
    preallocated_tmp = map((pre, r) -> pre ./ epsilonize(r), preallocated, reqs_tmp1)
    mins = nonzero_mins(preallocated_tmp, reqs_tmp1)
    m = minimum(mins)
    if m != Inf
        mm = [fill(m, size(rq)) for rq in reqs_tmp]
        perform_alloc!(mpools_tmp, reqs_tmp, preallocated_tmp, mpools_ratios, allocated,
                       allocated_per_mpool, mm, m, mins)
    end
    m != Inf
end

function alloc!(::Val{:prealloc_all}, mpools_tmp, reqs_tmp, preallocated, mpools_ratios,
                allocated, allocated_per_mpool, rows_mask)
    perform_alloc!(mpools_tmp, reqs_tmp, preallocated, mpools_ratios, allocated,
                   allocated_per_mpool, preallocated, 1, rows_mask)
    false
end

function pool_update_alloc!(pool :: Pool{T,N}, reqs, allocated, allocated_per_pool, indices) where {T <: Real, N}
    # println("pool_update_alloc!: \npool:$pool\nreqs:$reqs\nindices: $indices\nallocated: $allocated")
    allocated_per_pool[indices, :] .= reqs[indices, :] .* pool.x'
    allocated[indices, :] .+= allocated_per_pool[indices, :]
end

function pool_update_alloc!(pool :: Pool{T,N}, reqs, allocated, allocated_per_pool, indices) where {T <: Condition, N}
    allocated_per_pool[indices, :] .= reqs[indices, :] .* system.cache[:cond_cache][pool.label]
    allocated[indices, :] .+= allocated_per_pool[indices, :]
end

function copy_agent_row!(pool, allocated, allocated_per_pool, reqs, row_idx)
    for (i, r) in enumerate(reqs[row_idx, :])
        if r != 0
            append!(allocated[row_idx, i], pool.x[i])
            append!(allocated_per_pool[row_idx, i], pool.x[i])
        end
    end
end

function pool_update_alloc!(pool :: Pool{T,N}, reqs, allocated, allocated_per_pool, indices) where {T <: Agents, N}
    foreach(row_idx ->
            copy_agent_row!(pool, allocated, allocated_per_pool, reqs, row_idx),
            indices)
end

function pool_update_alloc!(pool :: Pool{Array{T,NN},N}, reqs, allocated, allocated_per_pool, indices) where {T <: Number, N, NN}
    if !iszero(reqs)
        foreach(indices) do idx
            foreach(enumerate(pool.x)) do (i, p)
                if reqs[idx, i]
                    copy!(allocated_per_pool[idx, i], p)
                end
            end
            foreach(range(1, length = length(pool.x))) do i
                allocated[idx, i] += allocated_per_pool[idx, i]
            end
        end
    end
end

function mpool_update_alloc!(mpool, allocated, alloc_per_mpool, reqs, ratios, indices)
    foreach((p, rs, al, al_per_pool, r) -> pool_update_alloc!(p, rs .* (r .!= 0), al, al_per_pool, indices),
            mpool.x, reqs, allocated, alloc_per_mpool, ratios)
end

function alloc!(::Val{:alloc_all}, mpools_tmp, reqs_tmp, preallocated, mpools_ratios,
                allocated, allocated_per_mpool, rows_mask)
    reqs_tmp1 = map(rq -> rq .* rows_mask, reqs_indicator(reqs_tmp))
    indices = findall(x -> x != 0, as_vector(rows_mask))
    foreach((mp, alloc_per_mp, ratios) ->
            mpool_update_alloc!(mp, allocated, alloc_per_mp, reqs_tmp1, ratios, indices),
            mpools_tmp.x, allocated_per_mpool, mpools_ratios)
    foreach(r -> r[indices,:] .= 0., reqs_tmp)
    false
end

# Allocated species from given multi pools per rule regarding weights
# Use allocation strategy regarding alloc types
function allocate(mpools, reqs, weights, mpools_mask, alloc_types, rows_alloc, system)
    reqs_tmp = deepcopy(reqs) # |> reqs_indicator
    mpools_tmp = all(a -> a == ALLOC_ALL_TYPE, alloc_types) ? mpools : deepcopy(mpools)
    # mpools_tmp = deepcopy(mpools)  # TODO: do we need deep copy everywhere ?
    allocated = empty_allocation(mpools.x[1], reqs)
    allocated_per_mpool = [empty_allocation(mp, reqs) for mp in mpools.x]
    cont_alloc = true
    while cont_alloc
        reqs_ind = reqs_indicator(reqs_tmp)
        cont_alloc = reduce(alloc_types, init = false) do ca, at
            rows_mask = rows_alloc .== at
            reqs_ind_tmp = map(rq -> Req(rq .* rows_mask, at), reqs_ind)
            preallocated, mpools_ratios = comp_prealloc(mpools_tmp, reqs_ind_tmp,
                                                        weights,
                                                        mpools_mask, system)
            alloc!(at, mpools_tmp, reqs_tmp, preallocated, mpools_ratios,
                   allocated, allocated_per_mpool, rows_mask) || ca
        end
    end
    allocated, mpools_tmp, allocated_per_mpool
end
