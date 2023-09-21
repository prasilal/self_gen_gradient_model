#module GridModels
import Random
#using ..DiceSets, ..Grids

abstract type AbstractGridModel end
abstract type AbstractGridModelState end
abstract type AbstractGridModelCfg end

struct GridModel{T, N, Ts, Tc} <: AbstractGridModel
    grid :: Grid{T, N}
    state :: Ts
    cfg :: Tc
end

### CPM

const BGID = 1
const BG_KIND = 1
const CELL_KIND_INVALID = typemin(Int)

mutable struct CPMState{Tdc, TRnd, N} <: AbstractGridModelState
    last_cell_id :: Int
    border_pixels :: DiceSet{Tdc, TRnd}
    neighbours :: Array{Int, N}
    cell_to_kind :: Array{Int, 1}
    cell_volume :: Array{Int, 1}
    constraints_states :: Dict{Symbol, Any}
    stat_state :: Dict{Symbol, Any}
    t :: Int
end

struct CPMCfg <: AbstractGridModelCfg
    params :: Dict{Symbol, Any}
    init_constraints_state :: Array{Function, 1}
    soft_constraints :: Array{Function, 1}
    hard_constraints :: Array{Function, 1}
    post_setpix_listeners :: Array{Function, 1}
    post_mcs_listeners :: Array{Function, 1}
    stats :: Dict{Symbol, Function}
end

function make_cpm_state(sizes :: NTuple{N, Int}, seed) where {N}
    Random.seed!(seed)
    CPMState(
        BGID,
        DiceSet{CartesianIndex{N}, Random.MersenneTwister}(Random.MersenneTwister(seed)),
        zeros(Int, sizes...),
        [BG_KIND],
        [prod(sizes)],
        Dict{Symbol, Any}(),
        Dict{Symbol, Any}(),
        0
    )
end

function make_empty_cpm_cfg(; stats = Dict{Symbol, Function}())
    CPMCfg(
        Dict{Symbol,Any}(),
        Function[],
        Function[],
        Function[],
        Function[],
        Function[],
        stats
    )
end


function empty_cfg()
    Dict(
        :params => Dict{Symbol, Any}(),
        :init_constraints_state => Function[],
        :soft_constraints => Function[],
        :hard_constraints => Function[],
        :post_setpix_listeners => Function[],
        :post_mcs_listeners => Function[],
        :stats => Dict{Symbol, Function}()
    )
end

function mk_cfg(; kwargs...)
    fnames = fieldnames(CPMCfg)

    fvals = reduce(kwargs |> collect, init = empty_cfg()) do fv, (k, v)
        append!(fv[k], v)
        fv
    end

    CPMCfg(map(k -> fvals[k], fnames)...)
end

function dict2cfg(d)
    fnames = fieldnames(CPMCfg)

    fvals = merge!(empty_cfg(), d)

    CPMCfg(map(k -> fvals[k], fnames)...)
end

params_cfg(; kwargs...) = dict2cfg(Dict(:params => Dict(kwargs...)))

join!(a1 :: Array{T,N}, a2 :: Array{T,N}) where {T, N} = append!(a1, a2)
join!(d1 :: Dict, d2 :: Dict) = merge!(d1, d2)

function Base.merge(cfgs :: CPMCfg...)
    names = fieldnames(CPMCfg)

    reduce(cfgs, init = mk_cfg()) do res_cfg, cfg
        reduce(names, init = res_cfg) do res_cfg, name
            join!(getproperty(res_cfg, name), getproperty(cfg, name))
            res_cfg
        end
    end
end

function make_empty_cpm(sizes :: Tuple, seed; stats = Dict{Symbol, Function}())
    GridModel(
        Grid(fill(BGID, sizes...), tuple(falses(length(sizes))...)),
        make_cpm_state(sizes, seed),
        make_empty_cpm_cfg(stats = stats)
    )
end

function make_preinit_cpm(sizes :: NTuple{N, T}, seed; cfg = CPMCfg(), is_torus = false) where {T <: Int,N}
    torus = is_torus ? trues(length(sizes)) : falses(length(sizes))
    cpm = GridModel(
        Grid(fill(BGID, sizes...), tuple(torus...)),
        make_cpm_state(sizes, seed),
        cfg
    )

    for init in cfg.init_constraints_state
        init(cpm)
    end

    cpm
end

""" Update border elements ({@link borderpixels}) after a successful copy attempt.
    @listens {setpixi} because borders change when a copy succeeds.
    @param {IndexCoordinate} idx - coordinate of pixel that has changed.
    @param {CellId} t_old - id of the cell the pixel belonged to before the copy.
    @param {CellId} t_new - id of the cell the pixel has changed into.
"""
function update_near_border!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                             idx :: Tdc, t_old :: T, t_new :: T) where {T, Tdc, N, Trnd}
    if t_old == t_new
        return
    end

    neighbours :: Array{Int, N} = model.state.neighbours
    grid = model.grid
    g :: Array{T,N} = grid.x
    border_pixels :: DiceSet{Tdc, Trnd} = model.state.border_pixels

    Ni :: Vector{Tdc} = neigh(moore_stencil(N), grid, idx)
    wasborder = neighbours[idx] > 0
    neighbours[idx] = 0

    for i in 1:length(Ni)
        ni = Ni[i]
        nt = g[ni]

        if nt != t_new
            neighbours[idx] += 1
        end

        if nt == t_old
            if neighbours[ni] == 0
                push!(border_pixels, ni)
            end
            neighbours[ni] += 1
        end

        if nt == t_new
            neighbours[ni] -= 1
            if neighbours[ni] == 0
                delete!(border_pixels, ni)
            end
        end
    end

    if !wasborder && neighbours[idx] > 0
        push!(border_pixels, idx)
    end

    if wasborder && neighbours[idx] == 0
        delete!(border_pixels, idx)
    end
end

""" Change the pixel at position i into {@link CellId} t.
    This method overrides {@link GridBasedModel#setpixi} because we want to
    add postSetpixListeners for all the constraints, to keep track of relevant information.

    See also {@link setpix} for a method working with {@link ArrayCoordinate}s.

    @param {IndexCoordinate} idx - coordinate of pixel to change.
    @param {CellId} t - cellid to change this pixel into.
"""
function setpix!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                 idx :: Tdc, t :: T) where {T, Tdc, N, Trnd}
    t_old = model.grid[idx]

    model.state.cell_volume[t_old] -= 1

    model.grid[idx] = t
    model.state.cell_volume[t] += 1

    update_near_border!(model, idx, t_old, t)

    for l in model.cfg.post_setpix_listeners
        l(model, idx, t_old, t)
    end

    model
end

""" Iterator returning non-background border pixels on the grid.
	  See {@link cellBorderPixelIndices} for a version returning pixels
	  by their {@link IndexCoordinate} instead of {@link ArrayCoordinate}.
	  @return {Pixel} for each pixel, return an array [p,v] where p are
		the pixel's array coordinates on the grid, and v its value.
"""
function cell_border_pixels(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    g = model.grid
    els = model.state.border_pixels.elements

    if length(els) == 0
        return Iterators.rest([])
    end

    i = els[1]
    imap = Iterators.accumulate(Iterators.rest(els),
                                init = (g[i], i)) do _, e
                                    (g[e], e)
                                end

    Iterators.filter(imap) do (v, k) v != BGID end
end

""" Initiate a new {@link CellId} for a cell of {@link CellKind} "kind", and create elements
    for this cell in the relevant arrays.
    @param {CellKind} kind - cellkind of the cell that has to be made.
    @return {CellId} of the new cell.
"""
function make_new_cell_id!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                           kind :: Int) where {T, Tdc, N, Trnd}
    model.state.last_cell_id += 1
    newid = model.state.last_cell_id

    push!(model.state.cell_volume, 0)
    push!(model.state.cell_to_kind, kind)

    newid
end

@inline function cell_kind(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                           id) where {T, Tdc, N, Trnd}
    model.state.cell_to_kind[id]
end

@inline function set_cell_kind!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                                id, kind) where {T, Tdc, N, Trnd}
    model.state.cell_to_kind[id] = kind
end

""" Determine whether copy attempt will succeed depending on deltaH (stochastic).
    @param {number} deltaH - energy change associated with the potential copy.
    @return {boolean} whether the copy attempt succeeds.
"""
function docopy(delta_h :: N1, T :: N2) where {N1 <: Real, N2 <: Real}
    if delta_h < 0
        return true
    end

    return rand() < exp(-delta_h / T)
end

""" Returns total change in hamiltonian for all registered soft constraints together.
   @param {IndexCoordinate} sourcei - coordinate of the source pixel that tries to copy.
   @param {IndexCoordinate} targeti - coordinate of the target pixel the source is trying
   to copy into.
   @param {CellId} src_type - cellid of the source pixel.
   @param {CellId} tgt_type - cellid of the target pixel.
   @return {number} the change in Hamiltonian for this copy attempt.
"""
function delta_h(model  :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                 sourcei :: Tdc, targeti :: Tdc,
                 src_type :: T, tgt_type :: T) where {T, Tdc, N, Trnd}
    r = 0.0
    for t :: Function in model.cfg.soft_constraints
        r += t(model, sourcei, targeti, src_type, tgt_type)
    end

    r
end

function reset_stat_state!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},) where {T, Tdc, N, Trnd}
    model.state.stat_state = Dict{Symbol, Any}()
end


""" A time step in the CPM is a Monte Carlo step. This performs a
       number of copy attempts depending on grid size:

    1) Randomly sample one of the border pixels for the copy attempt.
    2) Compute the change in Hamiltonian for the suggested copy attempt.
    3) With a probability depending on this change, decline or accept the
       copy attempt and update the grid accordingly.
"""
function time_step!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    delta_t = 0.0

    border_pixels :: DiceSet{Tdc, Trnd}  = model.state.border_pixels
    grid :: Grid{T,N} = model.grid
    g :: Array{T,N} = grid.x
    stencil :: Stencil{N} = moore_stencil(N)
    temp = model.cfg.params[:T] # temperature

    # this loop tracks the number of copy attempts until one MCS is completed.
    while delta_t < 1.0
        # This is the expected time (in MCS) you would expect it to take to
        # randomly draw another border pixel.
        # delta_t += 1.0 / border_pixels.len
        delta_t += 1.0 / length(border_pixels.elements)

        # sample a random pixel that borders at least 1 cell of another type,
        # and pick a random neighbour of tha pixel
        tgt_i :: Tdc = sample(border_pixels)
        Ni :: Vector{Tdc} = neigh(stencil, grid,  tgt_i)
        src_i :: Tdc = Ni[rand(1:length(Ni))]

        src_type :: T = g[src_i]
        tgt_type :: T = g[tgt_i]

        # only compute the Hamiltonian if source and target belong to a different cell,
        # and do not allow a copy attempt into the stroma. Only continue if the copy attempt
        # would result in a viable cell.
        if tgt_type != src_type
            ok = true

            for h in model.cfg.hard_constraints
                if !h(model, src_i, tgt_i, src_type, tgt_type)
                    ok = false
                    break
                end
            end

            if ok
                hamiltonian = delta_h(model, src_i, tgt_i, src_type, tgt_type)
                # probabilistic success of copy attempt
                if docopy(hamiltonian, temp)
                    setpix!(model, tgt_i, src_type)
                end
            end
        end
    end

    reset_stat_state!(model)
    model.state.t += 1

    for l in model.cfg.post_mcs_listeners
        l(model)
    end
end

""" Compute a statistic on this model. Stats are
	  cached because many stats use each other; this prevents that 'expensive' stats are
	  computed twice.
	  @param {Stat} s - the stat to compute.
	  @return {anything} - the value of the computed stat. This is often a {@link CellObject}
	  or a {@link CellArrayObject}.
"""
function get_stat(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                  name) where {T, Tdc, N, Trnd}
    cache = model.state.stat_state

    if !haskey(cache, name)
        cache[name] = model.cfg.stats[name](model)
    end

    cache[name]
end

""" Get volume of the cell with {@link CellId}
	  @param {CellId} t - id of the cell to get volume of.
	  @return {number} the cell's current volume.
"""
@inline function get_volume(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                    t) where {T, Tdc, N, Trnd}
    model.state.cell_volume[t]
end

function cell_ids(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},) where {T, Tdc, N, Trnd}
    s = model.state
    filter(x -> x != BGID && s.cell_to_kind[x] != CELL_KIND_INVALID && s.cell_volume[x] != 0,
           keys(s.cell_volume))
end

function cell_ids_all(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},) where {T, Tdc, N, Trnd}
    keys(model.state.cell_volume)
end

function kill_cell!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                    cellid) where {T, Tdc, N, Trnd}
    cp = get_stat(model, :pixels_by_cell)
    cpi = cp[cellid]

    for p in cpi
        setpix!(model, p, BGID)
    end

    model.state.cell_to_kind[cellid] = CELL_KIND_INVALID
    model.state.cell_volume[cellid] = 0

    cache = model.state.stat_state
    if haskey(cache[:pixels_by_cell], cellid)
        delete!(cache[:pixels_by_cell], cellid)
    end
end

""" Seed a new cell at a random position. Return 0 if failed, ID of new cell otherwise.
	  * Try a specified number of times, then give up if grid is too full.
	  * The first cell will always be seeded at the midpoint of the grid.

	  See also {@link seedCellAt} to seed a cell on a predefined position.

	  @param {CellKind} kind - what kind of cell should be seeded? This determines the CPM
	  parameters that will be used for that cell.
	  @param {number} [max_attempts = 10000] - number of tries allowed. The method will
	  attempt to seed a cell at a random position, but this will fail if the position is
	  already occupied. After max_attempts fails, it will not try again. This can happen
	  if the grid is very full.
	  @return {CellId} - the {@link CellId} of the newly seeded cell, or 0 if the seeding
	  has failed.
"""
function seed_cell!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                    kind; max_attempts = 10000) where {T, Tdc, N, Trnd}
    grid = model.grid
    s = size(grid)
    p = CartesianIndex(midpoint(grid)...)

    while grid[p] != BGID && max_attempts > 0
        max_attempts -= 1
        p = CartesianIndex(map(n -> rand(1:n), s)...)
    end

    if grid[p] != BGID
        return 0
    end

    newid = make_new_cell_id!(model, kind)
    setpix!(model, p, newid)
    newid
end

""" Seed a new cell of celltype "kind" onto position "p".
		This succeeds regardless of whether there is already a cell there.

		See also {@link seedCell} to seed a cell on a random position.

		@param {CellKind} kind - what kind of cell should be seeded? This determines the CPM
		parameters that will be used for that cell.
		@param {ArrayCoordinate} p - position to seed the cell at.
		@return {CellId} - the {@link CellId} of the newly seeded cell, or 0 if the seeding
		has failed.
"""
function seed_cell_at!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                       kind, p; newid = nothing) where {T, Tdc, N, Trnd}
    newid = newid == nothing ? make_new_cell_id!(model, kind) : newid
    checkbounds(model.grid, p)
    setpix!(model, p, newid)
    newid
end

seed_cells!(model, kind, num) = map(i -> seed_cell!(model, kind), 1:num)

function seed_cells!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                     kinds) where {T, Tdc, N, Trnd}
    vcat(map(x -> seed_cells!(model, x[1] + 1, x[2]), enumerate(kinds))...)
end

""" Seed "n" cells of celltype "kind" at random points lying within a circle
		surrounding "center" with radius "radius".

		See also {@link seedCell} to seed a cell on a random position in the entire grid,
		and {@link seedCellAt} to seed a cell at a specific position.

		@param {CellKind} kind - what kind of cell should be seeded? This determines the CPM
		parameters that will be used for that cell.
		@param {number} n - the number of cells to seed (must be integer).
		@param {ArrayCoordinate} center - position on the grid where the center of the
		circle should be.
		@param {number} radius - the radius of the circle to seed cells in.
		@param {number} max_attempts - the maximum number of attempts to seed a cell.
		Seeding can fail if the randomly chosen position is outside the circle, or if
		there is already a cell there. After max_attempts the method will stop trying
		and throw an error.
"""
function seed_cells_in_circle!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                               kind, n, center, radius;
                               newid = nothing, max_attempts = 10000) where {T, Tdc, N, Trnd}
    r = radius + 0.5
    r2 = r*r
    cells = []
    grid = model.grid

    while n > 0
        max_attempts -= 1
        if max_attempts == 0
            return nothing
        end

        p = CartesianIndex(map(i -> rand(i-radius:i+radius), center.I))
        d = dist2(p.I, center.I)

        if d < r2 && grid[p] == BGID
            push!(cells, seed_cell_at!(model, kind, p, newid = newid))
            n -= 1
        end
    end

    cells
end

function make_box!(voxels, minc, maxc)
    rngs = map((mi, ma) -> mi:ma, minc, maxc)
    cis = CartesianIndices(tuple(rngs...))

    append!(voxels, cis |> collect)
end

""" Helper method to set an entire plane or line of pixels to a certain CellId at once.
	  The method takes an existing array of coordinates (which can be empty) and adds the pixels
	  of the specified plane to it. See {@link changeKind} for a method that sets such a
	  pixel set to a new value.

	  The plane is specified by fixing one coordinate (x,y,or z) to a fixed value, and
	  letting the others range from their min value 0 to their max value.

	  @param {ArrayCoordinate[]} voxels - Existing array of pixels; this can be empty [].
	  @param {number} coord - the dimension to fix the coordinate of: 0 = x, 1 = y, 2 = z.
	  @param {number} coordvalue - the value of the coordinate in the fixed dimension; location
	  of the plane.
	  @return {ArrayCoordinate[]} the updated array of pixels.
"""
function make_plane!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                     voxels, coord, coordvalue) where {T, Tdc, N, Trnd}
    s = size(model.grid)
    dims = length(s)
    minc = ones(Int, dims)
    maxc = [s...]

    minc[coord] = coordvalue
    maxc[coord] = coordvalue

    make_box!(voxels, minc, maxc)
end

""" Helper method that converts all pixels in a given array to a specific cellkind:
	  changes the pixels defined by voxels (array of coordinates p) into
	  the given cellkind.

	  @param {ArrayCoordinate[]} voxels - Array of pixels to change.
	  @param {CellKind} cellkind - cellkind to change these pixels into.
"""
function change_kind!(model, voxels, cellkind)
    newid = make_new_cell_id!(model, cellkind)
    for p in voxels
        setpix!(model, p, newid)
    end
    newid
end

function comp_cell_pca(cp :: Vector{Tdc}, N :: Int, com :: Vector{Float64}) where {Tdc}
    xs = Vector{Float64}(undef, N*length(cp))
    j = 1

    for p in cp
        tmp = p.I .- com
        @inbounds for i in 1:N
            xs[j] = tmp[i]
            j += 1
        end
    end
    xs = reshape(xs, N, length(cp))

    fit(PCA, xs)
end

""" Let cell "id" divide by splitting it along a line perpendicular to
	  its major axis.

	  @param {CellId} id - the id of the cell that needs to divide.
	  @return {CellId} the id of the newly generated daughter cell.
"""
function divide_cell!(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                      id) where {T, Tdc, N, Trnd}
    com :: Vector{Float64} = get_stat(model, :centroid)[id]
    cp :: Vector{Tdc} = get_stat(model, :pixels_by_cell)[id]
    # xs = map(c -> c.I - com, cp)
    # xs = reshape(reduce((m, i)->append!(m,i), xs, init = Float64[]), N, length(xs))

    pca = comp_cell_pca(cp, N, com)
    n = projection(pca)[:,1]

    nid = make_new_cell_id!(model, cell_kind(model, id))

    foreach(cp) do c
        x = [(c.I - com)...]
        side = dot(n, x)
        if side > 0
            setpix!(model, c, nid)
        end
    end

    reset_stat_state!(model)

    nid
end

#end # module GridModels
