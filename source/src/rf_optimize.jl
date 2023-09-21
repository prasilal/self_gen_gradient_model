using NearestNeighbors, Optim

function non_intersecting_shapes_potential(position_unit_pot, angle_unit_pot, collision_unit_pot,
                                           old_pos :: Matrix{T}, new_pos :: Matrix{T},
                                           shapes :: Vector, pos_ws :: Vector{T}, opt_pos :: Vector{T}) where {T <: Real}
    pos = reshape(opt_pos, size(old_pos))
    tree = BallTree(pos)

    pot_pos = position_unit_pot * wdist2(pos, new_pos, pos_ws)

    ds = norm_cols(new_pos - old_pos)
    ds_opt = norm_cols(pos - old_pos)
    n = min(num_non_zero_cols(ds), num_non_zero_cols(ds_opt))
    pot_angle = angle_unit_pot * (n - dot(ds, ds_opt))

    r = 2. * (map(get_bounding_sphere, shapes) |> maximum)

    pot_col = 0
    for i in 1:size(pos)[2]
        nbs = inrange(tree, pos[:,i], r, false)
        for j in nbs
            if i != j
                pot_col += collision_unit_pot * shapes_intersect_pot(shapes[i],  shapes[j], pos[:,i], pos[:,j])
            end
        end
    end

    pot_pos + pot_angle + pot_col
end

function minimize_shapes_intersections(shapes :: Vector{TS}, new_pos :: Matrix{T};
                                       position_unit_pot = 1., angle_unit_pot = 1.,
                                       collision_unit_pot = 1.) where {T <: Real, TS <: AbstractAgentShape}
    old_pos = @>> shapes map(get_pos) vec_of_vec_to_mtx
    ws = map(s -> get(get_attrs(s), :pos_weight, 1.0), shapes)

    f = partial(non_intersecting_shapes_potential, position_unit_pot, angle_unit_pot,
                collision_unit_pot, old_pos, new_pos, shapes, ws)
    res = optimize(f, reshape(new_pos, length(new_pos)))

    reshape(Optim.minimizer(res), size(new_pos))
end
