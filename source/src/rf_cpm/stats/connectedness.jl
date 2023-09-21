""" This Stat computes the 'connectedness' of cells on the grid. 
	  Keys are the {@link CellId} of all cells on the grid, corresponding values the
	  connectedness of the corresponding cell. 
"""

""" This method computes the connectedness of a specific cell. 
	  @return {number} the connectedness value of this cell, a number between 0 and 1.
"""
function connectedness_of_cell(model,  cellid)
    ccbc = get_stat(model, :connected_components_by_cell)

    v = ccbc[cellid]
    s = 0
    r = 0
    for comp in keys(v)
        volume = v[comp] |> length
        s += volume
    end

    for comp in keys(v)
        volume = v[comp] |> length
        r += (volume/s)^2
    end

    r
end

""" The compute method of Connectedness creates an object with 
	  connectedness of each cell on the grid.
	  @return {CellObject} object with for each cell on the grid
	  a connectedness value. 
"""
function connectedness_stat(model)
    connectedness = Dict()

    for ci in cell_ids(model)
        connectedness[ci] = connectedness_of_cell(model, ci)
    end

    connectedness
end
