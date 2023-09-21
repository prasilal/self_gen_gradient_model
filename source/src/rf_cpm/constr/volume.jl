""" Implements the volume constraint of Potts models.
    This constraint is typically used together with {@link Adhesion}.
"""

struct VolumeParams
    # strength of the constraint per cellkind.
    lambda_v :: Vector{Float64}

    # Target volume per cellkind.
    v :: Vector{Float64}
end

""" The volume constraint term of the Hamiltonian for the cell with id t.
	  @param {number} vgain - Use vgain=0 for energy of current volume, vgain=1
		  for energy if cell gains a pixel, and vgain = -1 for energy if cell loses a pixel.
	  @param {CellId} t - the cellid of the cell whose volume energy we are computing.
	  @return {number} the volume energy of this cell.
"""
function volconstraint_old(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                       vgain, t) where {T, Tdc, N, Trnd}
    params = model.cfg.params[:volume]
    k = cell_kind(model, t)
    l = params.lambda_v[k]

    if t == BGID || l == 0
        return 0
    end
    vdiff = params.v[k] - (get_volume(model, t) + vgain)

    return l * (vdiff ^ 2)
end

""" Method to compute the Hamiltonian for this constraint.
	  @param {IndexCoordinate} sourcei - coordinate of the source pixel that tries to copy.
	  @param {IndexCoordinate} targeti - coordinate of the target pixel the source is trying
	  to copy into.
	  @param {CellId} src_type - cellid of the source pixel.
	  @param {CellId} tgt_type - cellid of the target pixel.
	  @return {number} the change in Hamiltonian for this copy attempt and this constraint.
"""
function volume_delta_h_old(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                        src_i, tgt_i, src_type, tgt_type) where {T, Tdc, N, Trnd}
    # volume gain of src cell
    delta_h = volconstraint(model, 1, src_type) - volconstraint(model,  0, src_type)
    # volume loss of tgt cell
    delta_h += volconstraint(model, -1, tgt_type) - volconstraint(model, 0, tgt_type)

    delta_h
end

@inline function volconstraint(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                               lambda_v :: Vector{Float64}, v :: Vector{Float64},
                               dvgain :: Int, t :: T) where {T, Tdc, N, Trnd}
    k = cell_kind(model, t)
    l = lambda_v[k]

    if t == BGID || l == 0
        return 0
    end

    vdiff = v[k] - get_volume(model, t)

    l * dvgain * (dvgain + 2 * vdiff)
end

function volume_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                        src_i :: Tdc, tgt_i :: Tdc, src_type :: T, tgt_type :: T) where {T, Tdc, N, Trnd}
    params :: VolumeParams = model.cfg.params[:volume]
    lambda_v :: Vector{Float64} = params.lambda_v
    v :: Vector{Float64} = params.v
    # volume gain of src cell
    s_delta = volconstraint(model, lambda_v, v, -1, src_type)
    # volume loss of tgt cell
    t_delta = volconstraint(model, lambda_v, v, 1, tgt_type)

    s_delta + t_delta
end

const VolumeCfg = mk_cfg(
    soft_constraints = [volume_delta_h]
)
