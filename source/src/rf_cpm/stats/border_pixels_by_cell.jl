""" This Stat creates a {@link CellArrayObject} with the border cellpixels of each cell on the grid.
    Keys are the {@link CellId} of cells on the grid, corresponding values are arrays
    containing the pixels belonging to that cell. Coordinates are stored as {@link ArrayCoordinate}.
"""

""" The compute method of BorderPixelsByCell creates an object with the borderpixels of
    each cell on the grid.
    @returns {CellArrayObject} An object with a key for each cell on the grid, and as
    corresponding value an array with all the borderpixels of that
    cell. Each pixel is stored by its {@link ArrayCoordinate}.
"""
function border_pixels_by_cell_stat(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    cellborderpixels = Dict{T, Vector{Tdc}}()

    for (p,i) in cell_border_pixels(model)
        if !haskey(cellborderpixels, p)
            cellborderpixels[p] = [i]
        else
            push!(cellborderpixels[p], i)
        end
    end

    cellborderpixels
end
