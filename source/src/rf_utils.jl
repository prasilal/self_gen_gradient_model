function make_rule(type, inp, out, t, label)
    ReactiveRule{type}(ArrayPartition(inp...), ArrayPartition(out...), t, label)
end

function make_gen_rule(t1, t2, inp, out, t, label)
    GeneralReactiveRule{t1, t2}(ArrayPartition(inp...), ArrayPartition(out...), t, label)
end

function rf_step!(system :: System, dt;
                  rules_msk = nothing, cache_allocations = false,
                  reuse_allocations = false, perform_transition = true,
                  pools_mask = nothing)
    mpools = system.multi_pools
    sts = system.states
    trans_mtx = system.transitions

    reqs, ws, mpools_mask = system.cache[:reqs], system.cache[:reqs_weights], system.cache[:rules_mpools_mask]
    production_masks = system.cache[:production_masks]
    rules_by_state = system.cache[:rules_by_state]
    trans_mpools = system.cache[:transition_mpools]
    rules_state = system.cache[:rules_state]
    rows_alloc = system.cache[:rows_allocators]
    alloc_types = system.cache[:allocators_types]

    if !reuse_allocations && !haskey(system.cache, :tmp_allocations)
        allocated, _, allocated_per_mpool = allocate(mpools, reqs, ws, mpools_mask, alloc_types, rows_alloc, system)
        if cache_allocations
            system.cache[:tmp_allocations] = (allocated, allocated_per_mpool)
        end
    else
        allocated, allocated_per_mpool = system.cache[:tmp_allocations]
    end

    if !cache_allocations && !haskey(system.cache, :tmp_allocations)
        delete!(system.cache, :tmp_allocations)
    end

    apply!(mpools, rules_by_state, reqs, allocated, allocated_per_mpool,
           production_masks, rules_state, dt, system, rules_by_state_msk = rules_msk)

    if perform_transition
        trans_mtx = system.transitions(system, dt)

        transition!(mpools, trans_mtx, sts, trans_mpools, pools_mask = pools_mask)
    end
end

function rf_step!(system :: SystemOfSystems, dt; kwargs...)
    for s in system.systems
        rf_step!(s, dt; kwargs...)
    end
end

# TODO: deprecated only for backward compatibility - will be removed
const rf_step = rf_step!

function const_trans(trans_mtx)
    (_, _) -> trans_mtx
end

function ts_const_trans(trans_mtx)
    (_, dt) -> map(tm -> tm .* dt, trans_mtx)
end


function make_simple_mpool(type, init, label)
    p = Pool{type,ndims(init)}(init, label)
    ArrayPartition(p)
end

function make_trans_mtx_old(ms...)
    reshape(mapreduce((a...) -> [a...], append!, ms...), (length(ms), size(ms[1])...))
end

function make_trans_mtx_compact(ms)
    s = size(ms[1])
    l = length(ms)
    trans_mtx = zeros((l, s...))
    for idx in [(k, i, j)
                for k in range(1, length=l)
                for i in range(1, length=s[1])
                for j in range(1, length=s[2])]
        (k,i,j) = idx
        trans_mtx[idx...]=ms[k][i,j]
    end
    trans_mtx
end

function make_trans_mtx(ms...)
    make_trans_mtx_compact(ms)
end

function part_prod_to(mpool, part)
    map(p -> fill(part, length(p.x)), mpool.x)
end

function full_prod_to(mpool)
    map(p -> ones(length(p.x)), mpool.x)
end

function none_prod_to(mpool)
    map(p -> zeros(length(p.x)), mpool.x)
end

function even_prod_to(mpools...)
    p = 1. / length(mpools)
    map(mp -> part_prod_to(mp, p), mpools)
end

function weights_ones(mpool)
    ones(length(mpool))
end

function weights_zeros(mpool)
    zeros(length(mpool))
end

function single_state_trans_mtx()
    ones(1,1,1)
end

function agents_cp_alloc_to_used_produced(allocated, produced, used)
    foreach((p, a) -> append!.(p, a), produced, allocated)
    foreach((u, a) -> append!.(u, a), used, allocated)
end

function real_cp_alloc_to_used_produced(allocated, produced, used)
    foreach((p, a) -> p .= a, produced, allocated)
    foreach((u, a) -> u .= a, used, allocated)
end

function cp_pool(dst, src)
    foreach((p, a) -> p .= a, dst, src)
end

function zero_pool(pool)
    foreach(p -> p.= 0, pool)
end

@inline get_mpool_cont(multi_pools, i) = multi_pools.x[i].x

function make_sys_of_sys(systems)
    system = SystemOfSystems(systems[1].multi_pools, systems[1].species, systems, Dict())
    init_system!(system)
    system
end

function vec_of_vec_to_mtx(vs; dims = 1)
    tp = reduce(vs, init = Union{}) do tp, x
        Union{tp, eltype(x)}
    end

    if dims == 1
        m = Array{tp}(undef, (vs |> first |> length, vs |> length))
        for (i,v) in enumerate(vs)
            m[:, i] .= v
        end
    else
        m = Array{tp}(undef, (vs |> length, vs |> first |> length))
        for (i,v) in enumerate(vs)
            m[i, :] .= v
        end
    end

    m
end
