""" This constraint allows a "barrier" celltype from and into which copy attempts are forbidden.
"""

struct BarrierParams
    """ specify for each cellkind if it should be
 	      considered as a barrier. If so, all copy attempts into and from it are forbidden.
    """
    is_barrier
end

""" Method for hard constraints to compute whether the copy attempt fulfills the rule.
    model.cfg :barrier - specify for each cellkind if it should be
    considered as a barrier. If so, all copy attempts into and from it are forbidden.
    @param {IndexCoordinate} src_i - coordinate of the source pixel that tries to copy.
    @param {IndexCoordinate} tgt_i - coordinate of the target pixel the source is trying
    to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel.
    @return {boolean} whether the copy attempt satisfies the constraint.
"""
function barrier_fulfilled(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                           src_i, tgt_i, src_type, tgt_type ) where {T, Tdc, N, Trnd}
    is_barrier = model.cfg.params[:barrier].is_barrier

    if is_barrier[cell_kind(model, src_type)]
        return false
    end

    if is_barrier[cell_kind(model, tgt_type)]
        return false
    end

   true
end

const BarrierCfg = mk_cfg(
    hard_constraints = [barrier_fulfilled]
)
