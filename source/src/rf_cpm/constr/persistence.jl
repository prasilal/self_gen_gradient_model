""" This is a constraint in which each cell has a preferred direction of migration.
    This direction is only dependent on the cell, not on the specific pixel of a cell.

    This constraint works with torus as long as the field size is large enough.
"""

struct PersistenceParams
    # strength of the persistenceconstraint per cellkind.
    lambda_dir
    # the number of MCS over which the current direction is
    # determined. Eg if DELTA_T = 10 for a cellkind, then its current direction is given by
    # the vector from its centroid 10 MCS ago to its centroid now.
    delta_t
    # persistence per cellkind. If this is 1, its new
    # target direction is exactly equal to its current direction. If it is lower than 1,
    # angular noise is added accordingly.
    persist
end

struct PersistenceState
    halfsize
    cellcentroidlists
    celldirections
end

function init_persistence(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    model.state.constraints_states[:perimeter] = PersistenceState(
        div.(size(model.grid), 2),
        Dict(),
        Dict()
    )
end

""" Method to compute the Hamiltonian for this constraint.
    @param {IndexCoordinate} sourcei - coordinate of the source pixel that tries to copy.
    @param {IndexCoordinate} targeti - coordinate of the target pixel the source is trying
    to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel. This argument is not used by this
    method, but is supplied for consistency with other SoftConstraints. The CPM will always
    call this method supplying the tgt_type as fourth argument.
    @return {number} the change in Hamiltonian for this copy attempt and this constraint.
"""
function persistence_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                             sourcei, targeti, src_type, tgt_type) where {T, Tdc, N, Trnd}
    state = model.state.constraints_states[:perimeter]
    if src_type == BGID || !haskey(state.celldirections, src_type)
        return 0
    end

    torus = model.grid.torus
    b = state.celldirections[src_type]
    p1 = sourcei
    p2 = targeti
    a = []
    s = size(model.grid)
    for i in 1:length(p1)
        a[i] = p2[i]-p1[i]
        # Correct for torus if necessary
        if torus[i]
            if a[i] > state.halfsize[i]
                a[i] -= s[i]
            elseif a[i] < -state.halfsize[i]
                a[i] += s[i]
            end
        end
    end

    dp = dot(a, b)
    -dp
end

random_dir(n) = rand(Normal(), n) |> normalize

""" After each MCS, update the target direction of each cell based on its actual
    direction over the last {conf.DELTA_T[cellkind]} steps, and some angular noise
    depending on {conf.PERSIST[cellkind]}.
    @listens {CPM#timeStep} because when the CPM has finished an MCS, cells have new
    centroids and their direction must be updated.
"""
function persistence_post_mcs_listener(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    params = model.cfg.params[:persistence]
    state = model.state.constraints_states[:persistence]
    torus = model.grid.torus
    grid_size = size(model.grid)

    centroids = get_stat(model, :centroids)

    for t in cell_ids(model)
        k = cell_kind(model, t)
        ld = params.lambda_dir[k]
        dt = params.delta_t != nothing && haskey(params.delta_t, k) ? params.delta_t[k] : 10

        if ld == 0
            delete!(state.cellcentroidlists, t)
            delete!(state.celldirections, t)
            continue
        end

        if !haskey(state.cellcentroidlists, t)
            state.cellcentroidlists[t] = []
            state.celldirections[t] = random_dir(N)
        end

        ci = centroids[t]
        pushfirst!(state.cellcentroidlists[t], ci)

        if state.cellcentroidlists[t] |> length >= dt
            l = nothing
            while state.cellcentroidlists[t] |> length >= dt
                l = pop!(state.cellcentroidlists[t])
            end

            dx = l |> length |> zeros

            for j in 1:length(l)
                dx[j] = ci[j] - l[j]

                if torus[j]
                    if dx[j] > state.halfsize[j]
                        dx[j] -= grid_size[j]
                    elseif dx[j] < -state.halfsize[j]
                        dx[j] += grid_size[j]
                    end
                end
            end

            per = params.persist[k]
            if per < 1
                normalize!(dx)
                normalize!(state.celldirections[t])
                @. dx = (1-per)*dx + per*state.celldirections[t]
                @. dx = normalize(dx) * ld
                state.celldirections[t] .= dx
            end
        end
    end
end

const PersistenceCfg = mk_cfg(
    init_constraints_state = [init_persistence],
    soft_constraints = [persistence_delta_h],
    post_mcs_listeners = [persistence_post_mcs_listener]
)
