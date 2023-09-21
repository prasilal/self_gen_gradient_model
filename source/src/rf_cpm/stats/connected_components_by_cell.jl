""" This Stat creates an object with the connected components of each cell on the grid.
    Keys are the {@link CellId} of all cells on the grid, corresponding values are objects
    where each element is a connected component. Each element of that array contains
    the {@link ArrayCoordinate} for that pixel.
"""

""" This method computes the connected components of a specific cell.
    @param {CellId} cellid the unique cell id of the cell to get connected components of.
    @returns {object} object of cell connected components. These components in turn consist of the pixels
    (specified by {@link ArrayCoordinate}) belonging to that cell.
"""
function connected_components_of_cell(model, cellid)
    cbp = get_stat(model, :pixels_by_cell)

    connected_components_of(model, Set(cbp[cellid]))
end

""" Creates an object with connected components of the border of each cell on the grid.
    @return {CellObject} object with for each cell on the grid
    an object of components. These components in turn consist of the pixels
    (specified by {@link ArrayCoordinate}) belonging to that cell.
"""
function connected_components_by_cell_stat(model)
    components = Dict()

    for ci in cell_ids(model)
        components[ci] = connected_components_of_cell(model,  ci)
    end

    components
end
