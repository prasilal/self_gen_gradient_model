#module Vizs

vox_log(x) = log(1 + (x >= 0 ? x : 0))

max_log = vox_log ∘ maximum
min_log = vox_log ∘ minimum

_vox(grid, v) = v == nothing ? empty_vox(grid) : v

empty_vox(size :: Tuple) = fill(RGBA(0.,0.,0.,0.), size)
empty_vox(grid) = empty_vox(size(grid))

"""Use to color a grid according to its values. High values are colored in
	 a brighter color.
	 @param {Grid} [grid] - the grid to draw values for. If left
	 unspecified, the grid that was originally supplied to the Canvas
	 constructor is used.
	 @param {RGBA} [col] - the color to draw the chemokine in.
"""
function draw_field(grid :: Grid{T, N}, color :: RGBA{C}; vox = nothing) where {T, N, C}
    g = grid.x
    sc2v = (x,y) -> begin
        alpha = vox_log(x) / maxval
        RGBA(color.r*alpha + y.r*y.alpha,
             color.g*alpha + y.g*y.alpha,
             color.b*alpha+y.b*y.alpha,
             min(alpha + y.alpha,0.9))
    end

    vox = _vox(grid, vox)
    maxval = max(max_log(g), 1e-10)
    vox .= sc2v.(g,vox)

    vox
end

""" Use to color a grid according to its values. High values are colored in
	  a brighter color.
	  @param {Grid} [grid] - the grid to draw values for. If left
	  unspecified, the grid that was originally supplied to the Canvas
	  constructor is used.
	  @param {number} [nsteps = 10] - the number of contour lines to draw.
	  Contour lines are evenly spaced between the min and max log10 of the
	  chemokine.
	  @param {RGBA} [col] - the color to draw contours with.
"""
function draw_field_contour(grid :: Grid{T, N}, color :: RGBA{C}; nsteps = 10, vox = nothing) where {T, N, C}
    g = grid.x
    vox = _vox(grid, vox)

    maxval = max_log(g)
    minval = min_log(g)

    step = (maxval-minval)/nsteps

    if step == 0
        return vox
    end

    for v in minval:step:maxval
        for idx in eachindex(grid)
            pixelval = vox_log(g[idx])

            if abs(v - pixelval) < 0.05 * maxval
                below = false
                above = false
                for n in neigh(neumann_stencil(N), grid, idx)
                    nval = vox_log(g[n])
                    if nval < v
                        below = true
                    end
                    if nval >= v
                        above = true
                    end

                    if above && below
                        alpha = 0.7 * ((v - minval) / (maxval - minval)) + 0.3
                        vox[idx] = RGBA(color.r, color.g, color.b, alpha)
                        break
                    end
                end
            end
        end
    end

    vox
end

""" @desc Method for drawing the cell borders for a given cellkind in the
	  color specified in "col" (hex format). This function draws a line around
	  the cell (rather than coloring the outer pixels). If [kind] is negative,
	  simply draw all borders.

	  See {@link drawOnCellBorders} to color the outer pixels of the cell.

	  @param {CellKind} kind - Integer specifying the cellkind to color.
	  Should be a positive integer as 0 is reserved for the background.
	  @param {RGBA}  [col] - the color to use
"""
function draw_cell_borders(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                           kind, col :: RGBA{C}; vox = nothing) where {T, Tdc, N, Trnd, C}
    g = model.grid
    vox = _vox(g, vox)

    for x in cell_border_pixels(model)
        p_kind = cell_kind(model, x[1])
        p = x[2]

        if p_kind == kind
            pc = x[1]

            for n in neigh(neumann_stencil(N), g, p)
                if g[n] != pc
                    vox[p] = col
                end
            end
        end
    end

    vox
end

""" Color outer pixel of all cells of kind [kind] in col [col].
	  See {@link drawCellBorders} to actually draw around the cell rather than
	  coloring the outer pixels. If you're using this model on a CA,
	  {@link CellKind} is not defined and the parameter "kind" is instead
	  interpreted as {@link CellId}.

	  @param {CellKind} kind - Integer specifying the cellkind to color.
	  Should be a positive integer as 0 is reserved for the background.
	  @param {HexColor|function} col - Optional: hex code for the color to use.
	  If left unspecified, it gets the default value of black ("000000").
	  col can also be a function that returns a hex value for a cell id.
"""
function draw_on_cell_borders(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                              kind, col :: Function; vox = nothing) where {T, Tdc, N, Trnd, C}
    vox = _vox(model.grid, vox)

    for p in cell_border_pixels(model)
        p_kind = cell_kind(model, p[1])

        if kind == p_kind
            vox[p[2]] = col(p[1])
        end
    end

    vox
end

""" Draw all cells of cellkind "kind" in color col. This method is
	  meant for models of class {@link CPM}, where the {@link CellKind} is
	  defined. If you apply this method on a {@link CA} model, this method
	  will internally call {@link drawCellsOfId} by just supplying the
	  "kind" parameter as {@link CellId}.

	  @param {CellKind} kind - Integer specifying the cellkind to color.
	  Should be a positive integer as 0 is reserved for the background.
	  @param {RGBA} col - the color to use.
"""
function draw_cells(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                    kind, col :: RGBA{C}; vox = nothing ) where {T, Tdc, N, Trnd, C}
    vox = _vox(model.grid, vox)
    cellpixelsbyid = get_stat(model, :pixels_by_cell)

    for cid in keys(cellpixelsbyid)
        if cell_kind(model, cid) == kind
            vox[cellpixelsbyid[cid]] .= col
        end
    end

    vox
end

function draw_cells(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                    col :: Function; vox = nothing) where {T, Tdc, N, Trnd}
    vox = _vox(model.grid, vox)
    cellpixelsbyid = get_stat(model, :pixels_by_cell)

    for cid in keys(cellpixelsbyid)
        vox[cellpixelsbyid[cid]] .= col(cid)
    end

    vox
end

function draw_pixel_set(pixelarray :: Array{CartesianIndex{N}}, size, col :: RGBA{C}; vox = nothing) where {N, C}
    vox = _vox(size, vox)

    vox[p] .= col

    vox
end

function act_col(a)
    r = [0., 0., 0., 1.0]

    if a > 0.5
        r[1] = 1.0
        r[2] = 2. - 2.0*a
    else
        r[1] = 2.0 * a
        r[2] = 1.0
    end

    RGBA(r...)
end

""" Use to show activity values of the act model using a color gradient, for
	  cells in the grid of cellkind "kind". The constraint holding the activity
	  values can be supplied as an argument. Otherwise, the current CPM is
	  searched for the first registered activity constraint and that is then
	  used.

	  @param {CellKind} kind - Integer specifying the cellkind to color.
	  If negative, draw values for all cellkinds.
	  @param {Function} [col] - a function that returns a color for a number
	  in [0,1] as an array of red/green/blue values, for example, [255,0,0]
	  would be the color red. If unspecified, a green-to-red heatmap is used.
"""
function draw_activity_values(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                              kind;
                              col :: Function = act_col, vox = nothing) where {T, Tdc, N, Trnd}
    g = model.grid
    state = model.state.constraints_states[:activity]
    params = model.cfg.params[:activity]
    vox = _vox(model.grid, vox)

    for x in eachindex(g)
        cid = g[x]
        k = cid != BGID ? cell_kind(model, cid) : 0

        if k == kind
            a = pxact(state, x) / params.max_act[k]
            if a > 0
                vox[x] = col(a)
            end
        end
    end

    vox
end

#end # modele Vizs
