""" This module implements the activity constraint of Potts models published in:

    Niculescu I, Textor J, de Boer RJ (2015)
    Crawling and Gliding: A Computational Model for Shape-Driven Cell Migration.
    PLoS Comput Biol 11(10): e1004280.

    Pixels recently added to a cell get an "activity", which then declines with every MCS.
    Copy attempts from more active into less active pixels have a higher success rate,
    which puts a positive feedback on protrusive activity and leads to cell migration.

    This constraint is generally used together with {@link Adhesion}, {@link VolumeConstraint},
    and {@link PerimeterConstraint}.
    @see https://doi.org/10.1371/journal.pcbi.1004280
"""

struct ActivityParams
    """ Local mean activity be measured with an
        "arithmetic" or a "geometric" mean?
    """
    act_mean :: Val

    """ Strength of the activity constraint per cellkind.
    """
    lambda_act :: Vector{Float64}

    """ How long do pixels remember their activity? Given per cellkind.
    """
    max_act :: Vector{Float64}
end

""" Init constraint state

    model.state.constraints_states[:activity]:
    Activity of all cellpixels with a non-zero activity is stored in this object,
    with the {@link IndexCoordinate} of each pixel as key and its current activity as
    value. When the activity reaches 0, the pixel is removed from the object until it
    is added again.
"""
function init_activity(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    model.state.constraints_states[:activity] = Dict{CartesianIndex{N}, Float64}()
end

@inline pxact(state :: Dict{CartesianIndex{N}, Float64}, idx) where {N} = haskey(state, idx) ? state[idx] : 0.

""" Activity mean computation methods for arithmetic mean. It computes the mean activity
    of a pixel and all its neighbors belonging to the same cell.

    This method is generally called indirectly via {@link activityAt}, which is set
    based on the value of ACT_MEAN in the configuration object given to the constructor.

    @param {IndexCoordinate} i - pixel to evaluate local activity at.
    @return {number} the arithmetic mean of activities in this part of the cell.
"""
function activity_at(::Val{:arith}, model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                     idx) where {T, Tdc, N, Trnd}
    state :: Dict{Tdc, Float64} = model.state.constraints_states[:activity]
    grid = model.grid
    g :: Array{T,N} = grid.x
    t = g[idx]

    # no activity for background/stroma
    if t <= BGID
        return 0
    end

    nbs :: Vector{Tdc} = neigh(moore_stencil(N), grid, idx)

    # r activity summed
    r = pxact(state, idx)
    nof_nbs = 1

    for i in 1:length(nbs)
        n = nbs[i]
        tn = g[n]

        # a neighbor only contributes if it belongs to the same cell
        if tn == t
            r += pxact(state, n)
            nof_nbs += 1
        end
    end

    # average is summed r divided by num neighbors.
    r / nof_nbs
end

""" Activity mean computation methods for geometric mean. It computes the mean activity
    of a pixel and all its neighbors belonging to the same cell.

    This method is generally called indirectly via {@link activityAt}, which is set
    based on the value of ACT_MEAN in the configuration object given to the constructor.

    @param {IndexCoordinate} i - pixel to evaluate local activity at.
    @return {number} the geometric mean of activities in this part of the cell.
"""
function activity_at(::Val{:geom}, model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                     idx) where {T, Tdc, N, Trnd}
    state :: Dict{Tdc, Float64} = model.state.constraints_states[:activity]
    grid = model.grid
    g :: Array{T,N} = grid.x
    t = g[idx]

    # no activity for background/stroma
    if t <= BGID
        return 0
    end

    nbs :: Vector{Tdc} = neigh(moore_stencil(N), grid, idx)

    # r activity summed
    r = pxact(state, idx)
    nof_nbs = 1

    for i in 1:length(nbs)
        n = nbs[i]
        tn = g[n]

        # a neighbor only contributes if it belongs to the same cell
        # if it does and has activity 0, the product will also be zero so
        # we can already return.
        if tn == t
            pxa = pxact(state, n)
            if  pxa == 0
                return 0
            end
            r *= pxa
            nof_nbs += 1
        end
    end

    # Geometric mean computation.
    r ^ (1.0 / nof_nbs)
end

""" Method to compute the Hamiltonian for this constraint.
    @param {IndexCoordinate} sourcei - coordinate of the source pixel that tries to copy.
    @param {IndexCoordinate} targeti - coordinate of the target pixel the source is trying
    to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel.
    @return {number} the change in Hamiltonian for this copy attempt and this constraint.
"""
function activity_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                          sourcei, targeti, src_type, tgt_type) where {T, Tdc, N, Trnd}
    params :: ActivityParams = model.cfg.params[:activity]

    max_act = 0
    lambda_act = 0

    src_kind = cell_kind(model, src_type)
    tgt_kind = cell_kind(model, tgt_type)

    # use parameters for the source cell, unless that is the background.
    # In that case, use parameters of the target cell.
    if src_type != BGID
        max_act = params.max_act[src_kind]
        lambda_act = params.lambda_act[src_kind]
    else
        # special case: punishment for a copy attempt from background into
        # an active cell. This effectively means that the active cell retracts,
        # which is different from one cell pushing into another (active) cell.
        max_act = params.max_act[tgt_kind]
        lambda_act = params.lambda_act[tgt_kind]
    end

    if max_act == 0 || lambda_act == 0
        return 0
    end

    act_mean = params.act_mean
    lambda_act * (activity_at(act_mean, model, targeti)
                  - activity_at(act_mean, model, sourcei)) / max_act
end

""" The post_setpix_listeners of the activity constraint ensures that pixels are
    given their maximal activity when they are freshly added to a CPM.
    @listens {CPM#setpix!} because when a new pixel is set (which is determined in the CPM),
    its activity must change so that this class knows about the update.
    @param {IndexCoordinate} i - the coordinate of the pixel that is changed.
    @param {CellId} t_old - the cellid of this pixel before the copy
    @param {CellId} t - the cellid of this pixel after the copy.
"""
function activity_post_setpix_listener(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                       i, t_old, t) where {T, Tdc, N, Trnd}
    k = cell_kind(model, t)
    state :: Dict{Tdc, Float64} = model.state.constraints_states[:activity]
    params :: ActivityParams = model.cfg.params[:activity]
    state[i] = params.max_act[k]
end

""" The postMCSListener of the ActivityConstraint ensures that pixel activities
    decline with one after every MCS.
    @listens {CPM#timeStep} because when the CPM has finished an MCS, the activities must go down.
"""
function activity_post_mcs_listeners(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    state :: Dict{Tdc, Float64} = model.state.constraints_states[:activity]
    for k in keys(state)
        state[k] -= 1
        if state[k] <= 0
            delete!(state, k)
        end
    end
end

const ActivityCfg = mk_cfg(
    init_constraints_state = [init_activity],
    soft_constraints = [activity_delta_h],
    post_setpix_listeners = [activity_post_setpix_listener],
    post_mcs_listeners = [activity_post_mcs_listeners]
)
