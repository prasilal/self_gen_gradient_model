""" Implements bias of motion in the direction of a supplied "attraction point".
    This constraint computes the cosine of the angle alpha between the direction
    of the copy attempt (from source to target pixel), and the direction from the
    source to the attraction point. This cosine is 1 if these directions are
    aligned, 0 if they are perpendicular, and 1 if they are opposite.
    We take the negative (so that deltaH is negative for a copy attempt in the
    right direction), and modify the strength of this bias using the lambda
    parameter. The constraint only acts on copy attempts *from* the cell that
    is responding to the field; it does not take into account the target pixel
    (except for its location to determine the direction of the copy attempt).

    The current implementation works for torus grids as long as the grid size in
    each dimension is larger than a few pixels.
"""

struct AttractionPointParams
    """ Strength of the constraint per cellkind.
    """
    lambda_attraction_point

    """ Coordinate of the attraction point.
    """
    attraction_point
end

""" Method to compute the Hamiltonian for this constraint.
    @param {IndexCoordinate} src_i - coordinate of the source pixel that
    tries to copy.
    @param {IndexCoordinate} tgt_i - coordinate of the target pixel the
    source is trying to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel. This argument is
    not actually used but is given for consistency with other soft
    constraints; the CPM always calls this method with four arguments.
    @return {number} the change in Hamiltonian for this copy attempt and
    this constraint.
"""
function attraction_point_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                  src_i, tgt_i, src_type, tgt_type ) where {T, Tdc, N, Trnd}
    params = model.cfg.params[:attraction_point]

    # deltaH is only non-zero when the source pixel belongs to a cell with
    # an attraction point, so it does not act on copy attempts where the
    # background would invade the cell.

    l = params.lambda_attraction_point[cell_kind(model, src_type)]
    if l == 0
        return 0
    end

    # To assess whether the copy attempt lies in the direction of the
    # attraction point, we must take into account whether the grid has
    # wrapped boundaries (torus; see below).

    torus = model.grid.torus
    tgt = params.attraction_point[cell_kind(model, src_type)]
    p1 = src_i.I
    p2 = tgt_i.I

    # To bias a copy attempt p1 -> p2 in the direction of vector 'dir'.
    # r will contain the dot product of the copy attempt vector and the
    # vector pointing from the source pixel to the attraction point.
    # The copy attempt vector always has length one, but the vector to the
    # attraction point has a variable length that will be stored in ldir
    # (actually, we store the squared length).

    r = 0.0
    ldir = 0.0
    s = size(model.grid)

    for i in 1:length(s)
        # compute the distance between the target and the current position
        # in this dimension, and add it in squared form to the total.
        dir_i = tgt[i] - p1[i]
        ldir += dir_i * dir_i

        # similarly, the distance between the source and target pixel in this
        # dimension (direction of the copy attempt is from p1 to p2)
        dx = p2[i] - p1[i]

        # we may have to correct for torus if a copy attempt crosses the
        # boundary.
        si = s[i]
        if torus[i]
            # If distance is greater than half the grid size, correct the
            # coordinate.
            if dx > si / 2
                dx -= si
            elseif dx < -si/2
                dx += si
            end
        end

        # direction of the gradient; add contribution of the current
        # dimension to the dot product.
        r += dx * dir_i
    end

    # divide dot product by squared length of directional vector to obtain
    # cosine of the angle between the copy attempt direction and the
    # direction to the attraction point. This cosine is 1 if they are
    # perfectly aligned, 0 if they are perpendicular, and negative
    # if the directions are opposite. Since we want to reward copy attempts
    # in the right direction, deltaH is the negative of this (and
    # multiplied by the lambda weight factor).

    - r * l / sqrt(ldir)
end

const AttractionPointCfg = mk_cfg(
    soft_constraints = [attraction_point_delta_h]
)
