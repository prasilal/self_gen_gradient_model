""" Soft version of the {@link ConnectivityConstraint} which only checks local connectivity.
    @experimental
"""

struct SoftLocalConnectivityParams
    """ Strength of the penalty for breaking connectivity.
    """
    lambda_connectivity

    """ Should a Neumann (default) or Moore neighborhood be used to determine
	      whether the cell locally stays connected? The default is Neumann since the Moore neighborhood tends to
	      give artefacts. Also, LAMBDA should be much higher if the Moore neighborhood is used.
    """
    nbh_type
end

""" Method to compute the Hamiltonian for this constraint.
	  @param {IndexCoordinate} src_i - coordinate of the source pixel that tries to copy.
	  @param {IndexCoordinate} tgt_i - coordinate of the target pixel the source is trying
	  to copy into.
	  @param {CellId} src_type - cellid of the source pixel.
	  @param {CellId} tgt_type - cellid of the target pixel.
	  @return {number} the change in Hamiltonian for this copy attempt and this constraint.
"""
function soft_local_connectivity_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                         src_i, tgt_i, src_type, tgt_type) where {T, Tdc, N, Trnd}
    params = model.cfg.params[:soft_local_connectivity]
    lambda = params.lambda_connectivity[cell_kind(model, tgt_type)]

    if tgt_type != BGID && lambda > 0
        return lambda * check_connected_local(model, tgt_i, src_type, tgt_type, stencil = params.nbh_type)
    end

    return 0
end

const SoftLocalConnectivityCfg = mk_cfg(
    soft_constraints = [soft_local_connectivity_delta_h]
)
