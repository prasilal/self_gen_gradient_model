""" Version of the {@link connectivity} which only checks local connectivity.
    @experimental
"""

""" This method checks if the connectivity still holds after pixel tgt_i is
    changed from tgt_type to src_type.
    @param {IndexCoordinate} tgt_i - the pixel to change.
    @param {CellId} src_type - the new cell for this pixel.
    @param {CellId} tgt_type - the cell the pixel belonged to previously.
"""
function check_connected_local(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                               tgt_i, src_type, tgt_type; stencil = moore_stencil) where {T, Tdc, N, Trnd}
    grid = model.grid
    nbh = neigh(moore_stencil(N), grid, tgt_i)

    nbh_obj = Dict()

    for n in nbh
        if grid[n] == tgt_type
            nbh_obj[n] = true
        end
    end

    connected_components_of(model, nbh_obj, stencil = stencil) |> length == 1
end

""" Method for hard constraints to compute whether the copy attempt fulfills the rule.
    @param {IndexCoordinate} src_i - coordinate of the source pixel that tries to copy.
    @param {IndexCoordinate} tgt_i - coordinate of the target pixel the source is trying
    to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel.
    @return {boolean} whether the copy attempt satisfies the constraint.
"""
function local_connectivity_fulfilled(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                      src_i, tgt_i, src_type, tgt_type) where {T, Tdc, N, Trnd}
    params = model.cfg.params[:connectivity]
    # connectedness of src cell cannot change if it was connected in the first place.
    # connectedness of tgt cell
    if tgt_type != BGID && params.connected[cell_kind(model, tgt_type)]
        return check_connected_local(model, tgt_i, src_type, tgt_type)
    end

    return true
end

const LocalConnectivityCfg = mk_cfg(
    hard_constraints = [local_connectivity_fulfilled]
)
