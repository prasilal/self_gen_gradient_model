""" This Stat computes the bounding ball of a cell. When the cell resides on a torus, the
    centroid may be well outside the cell, and other stats may be preferable.

    !!! Assumption: cell pixels never extend for more than half the size of the grid.
    If this assumption does not hold, bounding balls may be computed wrongly.
"""

""" This method computes the bouding ball of a specific cell with id = <cellid>.
    The cellpixels object is given as an argument so that it only has to be requested
    once for all cells together.
    @param {CellId} cellid ID number of the cell to get centroid of.
    @param {CellArrayObject} cellpixels object produced by {@link PixelsByCell},
    where keys are the cellids
    of all non-background cells on the grid, and the corresponding value is an array
    of the pixels belonging to that cell specified by their {@link ArrayCoordinate}.
    @param {Dict{T, Vector{Float64}}} centroids for cells
    @return {ArrayCoordinate} the centroid of the current cell.
"""
function comp_bbal(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                   cellid :: T, cellpixels :: Dict{T, Vector{Tdc}},
                   com :: Dict{T, Vector{Float64}}) where {T, Tdc, N, Trnd}
    pca = comp_cell_pca(cellpixels[cellid], N, com[cellid])
    r = principalvars(pca)[1]

    (r, com[cellid])
end

""" Compute bounding ball for all cells on the grid.
    @return {CellObject} with an {@link ArrayCoordinate} of the baounding balls for each cell
    on the grid.
"""
function bounding_ball_stat(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    cellpixels :: Dict{T, Vector{Tdc}} = get_stat(model, :pixels_by_cell)
    com :: Dict{T, Vector{Float64}} = get_stat(model, :centroid)

    bbals = Dict{T, Tuple{Float64, Vector{Float64}}}()

    for cid in cell_ids(model)
        bbals[cid] = comp_bbal(model, cid, cellpixels, com)
    end

    bbals
end
