""" This constraint forbids that cells exceed or fall below a certain size range.
"""

struct HardVolumeRangeParams
    # minimum volume of each cellkind.
    lambda_vrange_min
    # maximum volume of each cellkind.
    lambda_vrange_max
end

""" Method for hard constraints to compute whether the copy attempt fulfills the rule.
     @param {IndexCoordinate} src_i - coordinate of the source pixel that tries to copy.
     @param {IndexCoordinate} tgt_i - coordinate of the target pixel the source is trying
     to copy into.
     @param {CellId} src_type - cellid of the source pixel.
     @param {CellId} tgt_type - cellid of the target pixel.
     @return {boolean} whether the copy attempt satisfies the constraint.
"""
function hard_vol_rng_fulfilled(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                src_i, tgt_i, src_type, tgt_type) where {T, Tdc, N, Trnd}
    params = model.cfg.params[:hard_volume_range]
    cell_volume = model.state.cell_volume

    # volume gain of src cell
    if src_type != BGID && (cell_volume[src_type] + 1) >
        params.lambda_vrange_max[cell_kind(model, src_type)]
            return false
    end

    # volume loss of tgt cell
    if tgt_type != BGID && (cell_volume[tgt_type] - 1) <
        params.lambda_vrange_min[cell_kind(model, tgt_type)]
        return false
    end

    true
end

const HardVolumeRangeCfg = mk_cfg(
    hard_constraints = [hard_vol_rng_fulfilled]
)
