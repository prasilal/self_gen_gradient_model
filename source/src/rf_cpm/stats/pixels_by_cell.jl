""" This Stat creates an object with the cellpixels of each cell on the grid. 
    Keys are the {@link CellId} of all cells on the grid, corresponding values are arrays
    containing the pixels belonging to that cell. Each element of that array contains
    the {@link ArrayCoordinate} for that pixel.
"""

""" The compute method of PixelsByCell creates an object with cellpixels of each
    cell on the grid.
    @return {CellArrayObject} object with for each cell on the grid
    an array of pixels (specified by {@link ArrayCoordinate}) belonging to that cell.
"""
function pixels_by_cell_stat(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    cellpixels = Dict{T, Vector{Tdc}}()

    grid = model.grid
    g :: Array{T, N} = grid.x
    @inbounds for i :: Tdc in eachindex(grid)
        p = g[i]
        if p != BGID
            if haskey(cellpixels, p)
                idxs :: Vector{Tdc} = cellpixels[p]
                push!(idxs, i)
            else
                cellpixels[p] = Tdc[i]
            end
        end
    end

    cellpixels
end
