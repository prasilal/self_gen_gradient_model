""" This class implements a constraint for cells moving up a chemotactic gradient.
    It checks the chemokine gradient in the direction of the attempted copy, and
    rewards copy attempts that go up the gradient. This effect is stronger when the
    gradient is steep. Copy attempts going to a lower chemokine value are punished.

    The grid with the chemokine must be supplied in configuration.
"""

struct ChemotaxisParams
    """ chemotactic sensitivity per cellkind.
    """
    lambda_ch

    """the chemotactic field where the chemokine lives.
    """
    ch_field :: T where {T <: AbstractGrid}
end

pixt(grid :: Grid{T,N}, idx) where {T,N} = grid[idx]
pixt(grid :: CoarserGrid{T, N}, idx) where {T,N} = nearby_val(grid, idx)

""" Method to compute the Hamiltonian for this constraint.
    @param {IndexCoordinate} sourcei - coordinate of the source pixel that tries to copy.
    @param {IndexCoordinate} targeti - coordinate of the target pixel the source is trying
    to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel. This argument is not actually
    used by this method, but is supplied for compatibility; the CPM will always call the
    deltaH method with all 4 arguments.
    @return {number} the change in Hamiltonian for this copy attempt and this constraint.
"""
function chemotaxis_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                             sourcei, targeti, src_type, tgt_type) where {T, Tdc, N, Trnd}
    energy = 0.0

    for params in model.cfg.params[:chemotaxis]
        f = params.ch_field

        delta = pixt(f, targeti) - pixt(f, sourcei)
        lambdachem = params.lambda_ch[cell_kind(model, src_type)]
        energy += -delta*lambdachem
    end

    energy
end

const ChemotaxisCfg = mk_cfg(
    soft_constraints = [chemotaxis_delta_h]
)
