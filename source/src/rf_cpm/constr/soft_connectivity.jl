""" This constraint encourages that cells stay 'connected' throughout any copy attempts.
    In contrast to the hard version of the {@link ConnectivityConstraint}, this version does
    not completely forbid copy attempts that break the cell connectivity, but punishes them
    through a positive term in the Hamiltonian.
    @experimental
"""

struct SoftConnectivityParams
    """ Should the cellkind be connected or not?
    """
    lambda_connectivity
end

""" Compute the 'connectivity' of a cell; a number between 0 and 1. If the cell
	  is completely connected, this returns 1. A cell split into many parts gets a
	  connectivity approaching zero. It also matters how the cell is split: splitting
	  the cell in two near-equal parts results in a lower connectivity than separating
	  one pixel from the rest of the cell.
	  @param {Array} components - an array of arrays (one array per connected component,
	  in which each entry is the {@link ArrayCoordinate} of a pixel belonging to that component).
	  @param {CellId} cellid - the cell these components belong to.
	  @return {number} connectivity of this cell.
"""
function connectivity(model, components, cellid)
    if components |> length <= 1
        return 1
    end

    state = model.state.constraints_states[:connectivity]
    v_tot = keys(state.borderpixelsbycell[cellid]) |> length

    ci = 0
    for c in components
        vc = length(c)
        ci += (vc/v_tot)^2
    end

    (1 - ci) ^ 2
end

""" This method checks the difference in connectivity when pixel tgt_i is changed from
	  tgt_type to src_type.
	  @param {IndexCoordinate} tgt_i - the pixel to change
	  @param {CellId} src_type - the new cell for this pixel.
	  @param {CellId} tgt_type - the cell the pixel belonged to previously.
	  @return {number} conndiff - the difference: connectivity_after - connectivity_before.
"""
function check_connected_soft(model, tgt_i, src_type, tgt_type)
    if check_connected_local(model, tgt_i, tgt_type, tgt_type)
        return 0
    end

    comp1 = connected_components_of_cell_border(model, tgt_type)
    conn1 = connectivity(model, comp1, tgt_type)

    update_border_pixels(model, tgt_i, tgt_type, src_type)
    comp = connected_components_of_cell_border(model, tgt_type)
    conn = connectivity(model, comp, tgt_type)

    conndiff = conn2 - conn1

    update_border_pixels(model, tgt_i, src_type, tgt_type)

    conndiff
end

""" Method to compute the Hamiltonian for this constraint.
	  @param {IndexCoordinate} src_i - coordinate of the source pixel that tries to copy.
	  @param {IndexCoordinate} tgt_i - coordinate of the target pixel the source is trying
	  to copy into.
	  @param {CellId} src_type - cellid of the source pixel.
	  @param {CellId} tgt_type - cellid of the target pixel.
	  @return {number} the change in Hamiltonian for this copy attempt and this constraint.
"""
function soft_connectivity_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                   src_i, tgt_i, src_type, tgt_type) where {T, Tdc, N, Trnd}
    params = model.cfg.params[:soft_connectivity]
    lambda = params.lambda_connectivity[cell_kind(model, tgt_type)]

    if tgt_type != BGID && lambda > 0
        return lambda * check_connected_soft(model, tgt_i, src_type, tgt_type)
    end

    return 0
end

const SoftConnectivityCfg = mk_cfg(
    soft_constraints = [soft_connectivity_delta_h],
    init_constraints_state = [init_connectivity],
    post_setpix_listeners = [post_setpix_listener]
)
