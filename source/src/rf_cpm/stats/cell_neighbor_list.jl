""" This stat computes a list of all cell ids of the cells that border to "cell" and
    belong to a different cellid, also giving the interface length for each contact.
    @experimental
"""

""" The getNeighborsOfCell method of CellNeighborList computes a list of all pixels
    that border to "cell" and belong to a different cellid.
    @param {CellId} cellid the unique cell id of the cell to get neighbors from.
    @param {CellArrayObject} cellborderpixels object produced by {@link BorderPixelsByCell}, with keys for each cellid
    and as corresponding value the border pixel indices of their pixels.
    @returns {CellObject} a dictionairy with keys = neighbor cell ids, and
    values = number of neighbor cellpixels at the border.
"""
function get_neighbors_of_cell(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                               cellid, cellborderpixels) where {T, Tdc, N, Trnd}
    grid = model.grid
    neigh_cell_amountborder = Dict()
    cbp = cellborderpixels[cellid]

    for cellpix in cbp
        neighbours_of_borderpixel_cell = neigh(moore_stencil(N), grid, cellpix)
        for neighborpix in neighbours_of_borderpixel_cell
            neighbor_id = grid[neighborpix]
            if neighbor_id != cellid
                neigh_cell_amountborder[neighbor_id] = haskey(neigh_cell_amountborder, neighbor_id) ?
                    neigh_cell_amountborder[neighbor_id] + 1 : 1

            end
        end
    end

    neigh_cell_amountborder
end

""" The compute method of CellNeighborList computes for each cell on the grid
    a list of all pixels at its border that belong to a different cellid.
    @returns {CellObject} a dictionairy with keys = cell ids, and
    values = an object produced by {@link getNeighborsOfCell} (which has keys for each
    neighboring cellid and values the number of contacting pixels for that cell).
"""
function cell_neighbor_list_stat(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    cellborderpixels = get_stat(model, :border_pixels_by_cell)

    neighborlist = Dict()

    for i in cell_ids(model)
        neighborlist[i] = get_neighbors_of_cell(model, i, cellborderpixels)
    end

    neighborlist
end
