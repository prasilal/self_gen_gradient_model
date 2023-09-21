""" Implements the adhesion constraint of Potts models.
    Each pair of neighboring pixels [n,m] gets a positive energy penalty deltaH if n and m
    do not belong to the same {@link CellId}.
"""

struct AdhesionParams
    """ J[n][m] gives the adhesion energy between a pixel of
	      {@link CellKind} n and a pixel of {@link CellKind} m. J[n][n] is only non-zero
	      when the pixels in question are of the same {@link CellKind}, but a different
	      {@link CellId}. Energies are given as non-negative numbers.
    """
    J :: Array{Float64, 2}
end

""" Returns the Hamiltonian around a pixel idx with cellid tp by checking all its
    neighbors that belong to a different cellid.
    @param {IndexCoordinate} idx - coordinate of the pixel to evaluate hamiltonian at.
    @param {CellId} tp - cellid of this pixel.
    @return {number} sum over all neighbors of adhesion energies (only non-zero for
    neighbors belonging to a different cellid).
"""
function adhesion_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                    idx, tp) where {T, Tdc, N, Trnd}
    J = model.cfg.params[:adhesion].J
    g = model.grid
    cell_kind = model.state.cell_to_kind

    r = 0.
    nbs = neigh(moore_stencil(N), g, idx)
    tp_kind = cell_kind[tp]

    for n in 1:length(nbs)
        tn = g[nbs[n]]
        if tn != tp
            r += J[cell_kind[tn], tp_kind]
        end
    end

    r
end

""" Method to compute the Hamiltonian for this constraint.
    @param {IndexCoordinate} sourcei - coordinate of the source pixel that tries to copy.
    @param {IndexCoordinate} targeti - coordinate of the target pixel the source is trying
    to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel.
    @return {number} the change in Hamiltonian for this copy attempt and this constraint.
"""
function adhesion_delta_h_old(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                              sourcei, targeti, src_type, tgt_type) where {T, Tdc, N, Trnd}
    adhesion_h(model, targeti, src_type) - adhesion_h(model, targeti, tgt_type)
end

function adhesion_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                          sourcei :: Tdc, targeti :: Tdc,
                          src_type :: T, tgt_type :: T) where {T, Tdc, N, Trnd}
    params ::  AdhesionParams = model.cfg.params[:adhesion]
    J :: Array{Float64, 2} = params.J
    grid = model.grid
    g :: Array{T, N} = grid.x
    cell_kind :: Array{Int, 1} = model.state.cell_to_kind

    rt = rs = 0.0
    nbs :: Vector{Tdc} = neigh(moore_stencil(N), grid, targeti)
    src_kind = cell_kind[src_type]
    tgt_kind = cell_kind[tgt_type]

    @inbounds for n in 1:length(nbs)
        tn :: Int = g[nbs[n]]
        new_kind = cell_kind[tn]
        if tn != src_type
            rs += J[new_kind, src_kind]
        end
        if tn != tgt_type
            rt += J[new_kind, tgt_kind]
        end
    end

    rs - rt
end

const AdhesionCfg = mk_cfg(soft_constraints = [adhesion_delta_h])
