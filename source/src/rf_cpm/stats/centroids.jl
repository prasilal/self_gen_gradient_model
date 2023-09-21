""" This Stat computes the centroid of a cell. When the cell resides on a torus, the
    centroid may be well outside the cell, and other stats may be preferable (e.g.
    {@link CentroidsWithTorusCorrection}).

    !!! Assumption: cell pixels never extend for more than half the size of the grid.
    If this assumption does not hold, centroids may be computed wrongly.
"""

""" This method computes the centroid of a specific cell.
    @param {CellId} cellid the unique cell id of the cell to get centroid of.
    @param {CellArrayObject} cellpixels object produced by {@link PixelsByCell},
    with keys for each cellid
    and as corresponding value the pixel coordinates of their pixels.
    @returns {ArrayCoordinate} coordinate of the centroid.
"""
function compute_centroid_of_cell(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                  cellid :: T, cellpixels :: Dict{T, Vector{Tdc}}) where {T, Tdc, N, Trnd}
    pixels :: Vector{Tdc} = cellpixels[cellid]

    # cvec will contain the x, y, (z) coordinate of the centroid.
    # Loop over the dimensions to compute each element separately.
    cvec = zeros(ndims(model.grid))

    @inbounds for dim in 1:ndims(model.grid)
        mi = 0.0

        # Loop over the pixels;
        # compute mean position per dimension with online algorithm
        @inbounds for j in 1:length(pixels)
            # Check distance of current pixel to the accumulated mean in this dim.
            p :: Tdc = pixels[j]
            dx = p.I[dim] - mi

            # Update the mean with the appropriate weight.
            mi += dx/j
        end
        # Vector the mean position in the cvec vector.
        cvec[dim] = mi
    end

    cvec
end

""" This method computes the centroid of a specific cell with id = <cellid>.
    The cellpixels object is given as an argument so that it only has to be requested
    once for all cells together.
    @param {CellId} cellid ID number of the cell to get centroid of.
    @param {CellArrayObject} cellpixels object produced by {@link PixelsByCell},
    where keys are the cellids
    of all non-background cells on the grid, and the corresponding value is an array
    of the pixels belonging to that cell specified by their {@link ArrayCoordinate}.
    @return {ArrayCoordinate} the centroid of the current cell.
"""
function compute_centroid_of_cell_torus(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                        cellid :: T, cellpixels :: Dict{T, Vector{Tdc}}) where {T, Tdc, N, Trnd}
    torus = model.grid.torus
    pixels :: Vector{Tdc} = cellpixels[cellid]
    s = size(model.grid)
    hs = s ./ 2

    # cvec will contain the x, y, (z) coordinate of the centroid.
    # Loop over the dimensions to compute each element separately.
    cvec = zeros(ndims(model.grid))

    @inbounds for dim in 1:ndims(model.grid)
        mi = 0.0
        hsi = hs[dim]
        si = s[dim]

        # Loop over the pixels;
        # compute mean position per dimension with online algorithm
        @inbounds for j in 1:length(pixels)
            # Check distance of current pixel to the accumulated mean in this dim.
            # Check if this distance is greater than half the grid size in this
            # dimension; if so, this indicates that the cell has moved to the
            # other end of the grid because of the torus. Note that this only
            # holds AFTER the first pixel (so for j > 0), when we actually have
            # an idea of where the cell is.
            p :: Tdc = pixels[j]
            dx = p.I[dim] - mi

            if torus[dim] && j > 1
                if dx > hsi
                    dx -= si
                elseif dx < -hsi
                    dx += si
                end
            end

            # Update the mean with the appropriate weight.
            mi += dx/j
        end

        #  Correct the final position so that it falls in the current grid.
        #  (Because of the torus, it can happen to get a centroid at eg x = -1. )
        if mi < 0
            mi += si
        elseif mi > si
            mi -= si
        end

        # Vector the mean position in the cvec vector.
        cvec[dim] = mi
    end

    cvec
end

""" Compute centroids for all cells on the grid.
    @return {CellObject} with an {@link ArrayCoordinate} of the centroid for each cell
    on the grid (see {@link computeCentroidOfCell}).
"""
function centroid_stat(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    comp_centroid = all(x -> x == false, model.grid.torus) ?
        compute_centroid_of_cell : compute_centroid_of_cell_torus

    cellpixels :: Dict{T, Vector{Tdc}} = get_stat(model, :pixels_by_cell)

    centroids = Dict{T, Vector{Float64}}()

    for cid in cell_ids(model)
        centroids[cid] = comp_centroid(model, cid, cellpixels)
    end

    centroids
end
