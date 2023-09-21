""" This constraint enforces that cells stay 'connected' throughout any copy attempts.
    Copy attempts that break the cell into two parts are therefore forbidden. To speed things
    up, this constraint only checks if the borderpixels of the cells stay connected.

    @experimental
"""

struct ConnectivityParams
    """ should the cellkind be connected or not?
    """
    connected
end

struct ConnectivityState
    borderpixelsbycell
    neighbours
end

function init_connectivity(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    model.state.constraints_states[:connectivity] = ConnectivityState(
        Dict(),
        zeros(size(model.grid))
    )
end

""" Update the borderpixels when pixel i changes from t_old into t_new.
    @param {IndexCoordinate} i - the pixel to change
    @param {CellId} t_old - the cell the pixel belonged to previously
    @param {CellId} t_new - the cell the pixel belongs to now.
"""
function update_border_pixels(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                              i, t_old, t_new; state_key = :connectivity) where {T, Tdc, N, Trnd}
    state = model.state.constraints_states[state_key]
    grid = model.grid

    if t_old == t_new
        return
    end

    if !haskey(state.borderpixelsbycell, t_new)
        state.borderpixelsbycell[t_new] = Set()
    end

    nbs = neigh(moore_stencil(N), grid, i)
    wasborder = state.neighbours[i] > 0
    # current neighbors of pixel i, set counter to zero and loop over nbh.
    state.neighbours[i] = 0

    for ni in nbs
        nt = grid[ni]

        # If type is not the t_new of pixel i, nbi++ because the neighbor belongs
        # to a different cell.
        if nt != t_new
            state.neighbours[i] += 1
        end

        # If neighbor type is t_old, the border of t_old may have to be adjusted.
        # It gets an extra neighbor because the current pixel becomes t_new.
        if nt == t_old
            # If this wasn't a borderpixel of t_old, it now becomes one because
            # it has a neighbor belonging to t_new
            if state.neighbours[ni] == 0
                if !haskey(state.borderpixelsbycell, t_old)
                    state.borderpixelsbycell[t_old] = Set()
                end

                push!(state.borderpixelsbycell[t_old], ni)
            end
            state.neighbours[ni] += 1
        end

        # If type is t_new, the neighbor may no longer be a borderpixel
        if nt == t_new
            state.neighbours[ni] -= 1
            if (state.neighbours[ni] == 0) && (ni in state.borderpixelsbycell[t_new])
                delete!(state.borderpixelsbycell[t_new], ni)
            end
        end
    end

    # Case 1:
    # If current pixel is a borderpixel, add it to those of the current cell.
    if state.neighbours[i] > 0
        push!(state.borderpixelsbycell[t_new], i)
    end

    # Case 2:
    # Current pixel was a borderpixel. Remove from the old cell.
    if wasborder
        # It was a borderpixel from the old cell, but no longer belongs to that cell.
        if i in state.borderpixelsbycell[t_old]
            delete!(state.borderpixelsbycell[t_old], i)
        end
    end
end

function post_setpix_listener(model, i, t_old, t)
		update_border_pixels(model, i, t_old, t)
end

function label_component(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                         pixels, visited, pixelobject, stencil, seed, k) where {T, Tdc, N, Trnd}
    grid = model.grid

    q = [seed]
    cellid = grid[seed]

    push!(visited, seed)
    push!(pixels, [])

    while length(q) > 0
        e = pop!(q)
        push!(pixels[k], e)

        ne = neigh(stencil(N), grid, e)
        for n in ne
            if grid[n] == cellid &&
                !(n in visited) && haskey(pixelobject, n)
                push!(q, n)
                push!(visited, n)
            end
        end
    end
end

""" Get the connected components of a set of pixels.
    @param {object} pixelobject - an object with as keys the {@link IndexCoordinate}s of the pixels to check.
    @return {object} an array with an element for every connected component, which is in
    turn an array of the {@link ArrayCoordinate}s of the pixels belonging to that component.
"""
function connected_components_of(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                 pixelobject; stencil = moore_stencil) where {T, Tdc, N, Trnd}
    cbpi = keys(pixelobject)

    visited = Set()
    k = 1
    pixels = []
    label_component = partial(label_component, model, pixels, visited, pixelobject, stencil)

    for pi in cbpi
        if !(pi in visited)
            label_component(pi, k)
            k +=  1
        end
    end

    pixels
end


""" Get the connected components of the borderpixels of the current cell.
    @param {CellId} cellid - cell to check the connected components of.
    @return {object} an array with an element for every connected component, which is in
    turn an array of the {@link ArrayCoordinate}s of the pixels belonging to that component.
"""
function connected_components_of_cell_border(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                             cellid; state_key = :connectivity) where {T, Tdc, N, Trnd}
    state = model.state.constraints_states[state_key]

    if !haskey(state.borderpixelsbycell, cellid)
        return []
    end

    connected_components_of(state.borderpixelsbycell[cellid])
end

""" To speed things up: first check if a pixel change disrupts the local connectivity
    in its direct neighborhood. If local connectivity is not disrupted, we don't have to
    check global connectivity at all. This currently only works in 2D, so it returns
    false for 3D (ensuring that connectivity is checked globally).
    @param {IndexCoordinate} tgt_i - the pixel to change
    @param {CellId} tgt_type - the cell the pixel belonged to before the copy attempt.
    @return {boolean} does the local neighborhood remain connected if this pixel changes?
"""
function local_connected(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                        tgt_i, tgt_type) where {T, Tdc, N, Trnd}
    false
end

function local_connected(model :: GridModel{T, 2, CPMState{Tdc, Trnd, 2}, CPMCfg},
                        tgt_i, tgt_type) where {T, Tdc, Trnd}
    grid = model.grid
    neighbors = 0
    for i in neigh(neumann_stencil(2), model, tgt_i)
        if grid[i] != tgt_type
            neighbors += 1
        end
    end

    if neighbors >= 2
        false
    end

    true
end

""" This method checks if the connectivity still holds after pixel tgt_i is changed from
    tgt_type to src_type.
    @param {IndexCoordinate} tgt_i - the pixel to change
    @param {CellId} src_type - the new cell for this pixel.
    @param {CellId} tgt_type - the cell the pixel belonged to previously.
"""
function check_connected(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                         tgt_i, src_type, tgt_type) where {T, Tdc, N, Trnd}
    # If local connectivity is preserved, global connectivity holds too.
    if local_connected(model, tgt_i, tgt_type )
        return true
    end

    # Otherwise, check connected components of the cell border. Before the copy attempt:
    comp1 = connected_components_of_cell_border(model, tgt_type)
    length_before = length(comp1)

    # Update the borderpixels as if the change occurs
    update_border_pixels(model, tgt_i, tgt_type, src_type)
    comp = connected_components_of_cell_border(model, tgt_type)
    length_after = length(comp)

    # The src pixels copies its type, so the cell src_type gains a pixel. This
    # pixel is by definition connected because the copy happens from a neighbor.
    # So we only have to check if tgt_type remains connected
    connected = true
    if( length_after > length_before )
        connected = false
    end

    # Change borderpixels back because the copy attempt hasn't actually gone through yet.
    update_border_pixels(model, tgt_i, src_type, tgt_type)

    connected
end

""" Method for hard constraints to compute whether the copy attempt fulfills the rule.
    @param {IndexCoordinate} src_i - coordinate of the source pixel that tries to copy.
    @param {IndexCoordinate} tgt_i - coordinate of the target pixel the source is trying
    to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel.
    @return {boolean} whether the copy attempt satisfies the constraint.
"""
function connectivity_fulfilled(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                src_i, tgt_i, src_type, tgt_type) where {T, Tdc, N, Trnd}
    params = model.cfg.params[:connectivity]
    # connectedness of src cell cannot change if it was connected in the first place.
    # connectedness of tgt cell
    if tgt_type != BGID && params.connected[cell_kind(model, tgt_type)]
        check_connected(model, tgt_i, src_type, tgt_type)
    end

    return true
end

const ConnectivityCfg = mk_cfg(
    hard_constraints = [connectivity_fulfilled],
    init_constraints_state = [init_connectivity],
    post_setpix_listeners = [post_setpix_listener]
)
