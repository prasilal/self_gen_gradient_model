""" Implements the perimeter constraint of Potts models.
    A cell's "perimeter" is the number over all its borderpixels of the number of
    neighbors that do not belong to the cell itself.

    This constraint is typically used together with {@link Adhesion} and {@VolumeConstraint}.
"""

struct PerimeterParams
    """ strength of the perimeter constraint per cellkind.
    """
    lambda_p :: Vector{Float64}

    """ Target perimeter per cellkind.
    """
    p :: Vector{Float64}
end

function init_perimeter(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    grid = model.grid
    neighbours = model.state.neighbours
    cellperimeters = Dict{Int, Int}()
    model.state.constraints_states[:perimeter] = cellperimeters

    for bp in model.state.border_pixels.elements
        cid = grid[bp]
        if cid != BGID && !haskey(cellperimeters, cid)
            cellperimeters[cid] = 0
        end

        if cid != BGID
            cellperimeters[cid] += neighbours[bp]
        end
    end
end

""" The postSetpixListener of the PerimeterConstraint ensures that cell
    perimeters are updated after each copy in the CPM.
    @listens {CPM#setpixi} because when a new pixel is set (which is
      determined in the CPM), some of the cell perimeters will change.
    @param {IndexCoordinate} i - the coordinate of the pixel that is changed.
    @param {CellId} t_old - the cellid of this pixel before the copy
    @param {CellId} t_new - the cellid of this pixel after the copy.
"""
function perimeter_post_setpix_listener(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                        i, t_old, t_new) where {T, Tdc, N, Trnd}
    if t_old == t_new
        return
    end

    grid = model.grid
    g :: Array{T,N} = grid.x
    cellperimeters :: Dict{Int, Int} = model.state.constraints_states[:perimeter]
    # Neighborhood of the pixel that changes
    Ni :: Vector{CartesianIndex{N}} = neigh(moore_stencil(N), grid, i)
    # Keep track of perimeter before and after copy
    n_new = 0
    # Loop over the neighborhood.
    n_old = 0

    for j in 1:length(Ni)
        n = Ni[j]
        nt = g[n]

        # neighbors are added to the perimeter if they are
        # of a different cellID than the current pixel
        if nt != t_new
            n_new += 1
        end

        if nt != t_old
            n_old += 1
        end

        # if the neighbor is non-background, the perimeter
        # of the cell it belongs to may also have to be updated.
        if nt != BGID

            # if it was of t_old, its perimeter goes up because the
            # current pixel will no longer be t_old. This means it will
            # have a different type and start counting as perimeter.
            if nt == t_old
                cellperimeters[nt] += 1
            end
            # opposite if it is t_new.
            if nt == t_new
                cellperimeters[nt] -= 1
            end
        end
    end

    # update perimeters of the old and new type if they are non-background
    if t_old != BGID
        cellperimeters[t_old] -= n_old
    end
    if t_new != BGID
        if !haskey(cellperimeters, t_new)
            cellperimeters[t_new] = 0
        end
        cellperimeters[t_new] += n_new
    end
end

""" Method to compute the Hamiltonian for this constraint.
    @param {IndexCoordinate} sourcei - coordinate of the source pixel that
      tries to copy.
    @param {IndexCoordinate} targeti - coordinate of the target pixel the
      source is trying to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel.
    @return {number} the change in Hamiltonian for this copy attempt and
    this constraint.
"""
function perimeter_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                           sourcei, targeti, src_type, tgt_type) where {T, Tdc, N, Trnd}
    if src_type === tgt_type
        return 0.
    end

    grid = model.grid
    g :: Array{T,N} = grid.x
    params :: PerimeterParams = model.cfg.params[:perimeter]
    lambda_p :: Vector{Float64} = params.lambda_p
    p :: Vector{Float64} = params.p
    cellperimeters :: Dict{Int, Int} = model.state.constraints_states[:perimeter]

    ts = cell_kind(model, src_type)
    ls = lambda_p[ts]
    tt = cell_kind(model, tgt_type)
    lt = lambda_p[tt]

    if (ls <= 0) && (lt <= 0)
        return 0.
    end

    Ni :: Vector{CartesianIndex{N}} = neigh(moore_stencil(N), grid, targeti)
    pchange = Dict{Int, Int}()
    pchange[src_type] = 0
    pchange[tgt_type] = 0
    for i in 1:length(Ni)
        n = Ni[i]
        nt = g[n]

        if nt != src_type
            pchange[src_type] += 1
        end

        if nt != tgt_type
            pchange[tgt_type] -= 1
        end

        if nt == tgt_type
            pchange[nt] += 1
        end

        if nt == src_type
            pchange[nt] -= 1
        end
    end

    r = 0.0
    if ls > 0
        pt = p[ts]
        ps = cellperimeters[src_type]
        hnew = (ps+pchange[src_type])-pt
        hold = ps-pt
        r += ls*((hnew*hnew)-(hold*hold))
    end
    if lt > 0
        pt = p[tt]
        ps = cellperimeters[tgt_type]
        hnew = (ps+pchange[tgt_type])-pt
        hold = ps-pt
        r += lt*((hnew*hnew)-(hold*hold))
    end

    r
end

const PerimeterCfg = mk_cfg(
    init_constraints_state = [init_perimeter],
    post_setpix_listeners = [perimeter_post_setpix_listener],
    soft_constraints = [perimeter_delta_h]
)
