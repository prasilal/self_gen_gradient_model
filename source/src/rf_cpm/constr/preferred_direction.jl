""" Implements a global bias direction of motion.
	  This constraint computes the *unnormalized* dot product
	  between copy attempt vector and target direction vector.

	  Supply the target direction vector in normalized form, or
	  use the length of the vector to influence the strength
	  of this bias.

	  Works for torus grids, if they are "big enough".
"""

struct PreferredDirectionParams
    """ strength of the constraint per cellkind.
    """
    lambda_dir

    """ 'vector' with the preferred direction. This is
	      an array with the {@link ArrayCoordinate}s of the start and endpoints of this vector.
    """
    dir
end

""" Method to compute the Hamiltonian for this constraint.
	  @param {IndexCoordinate} src_i - coordinate of the source pixel that tries to copy.
	  @param {IndexCoordinate} tgt_i - coordinate of the target pixel the source is trying
	  to copy into.
	  @param {CellId} src_type - cellid of the source pixel.
	  @param {CellId} tgt_type - cellid of the target pixel. This argument is not actually
	  used but is given for consistency with other soft constraints; the CPM always calls
	  this method with four arguments.
	  @return {number} the change in Hamiltonian for this copy attempt and this constraint.
"""
function preferrred_direction_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                      sourcei, targeti, src_type, tgt_type) where {T, Tdc, N, Trnd}
    params = model.cfg.params[:preferrred_direction]
    l = params.lambda_dir[cell_kind(model, src_type)]
    if l == 0
        return 0
    end

    torus = model.grid.torus
    dir = params.dir[cell_kind(model, src_type)]
    p1 = sourcei
    p2 = targeti
    r = 0
    s = size(model.grid)
    for i in 1:length(p1)
        dx = p2[i] - p1[i]
        if torus[i]
            if dx > s[i]/2
                dx -= s[i]
            elseif dx < -si/2
                dx += s[i]
            end
        end
        r += dx * dir[i]
    end

    return - r * l
end

const PreferredDirectionCfg = mk_cfg(
    soft_constraints = [preferrred_direction_delta_h]
)
