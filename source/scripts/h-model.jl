using Colors
using GLMakie
using MLStyle
using Random
using Distributions

include("../src/rf.jl")
include("../src/rf_plot.jl")
include("../src/rf_cpm.jl")
include("../src/rf_cpm/Vizs.jl")

## Model

const BG_TYPE = BG_KIND
const CELL_TYPE_DEAD = BG_TYPE + 1
const CELL_TYPES_START = CELL_TYPE_DEAD + 1
const CELL_TYPES_OFFSET = CELL_TYPE_DEAD

mutable struct Cell
    id :: Int
    receptors :: Vector{Float64}
    state :: Dict{Symbol, Any}
end

mutable struct Sim{N}
    cells :: Vector{Cell}
    vaxes :: Vector{Grid{Float64, N}}
    model :: GridModel
    cfg :: Dict{Symbol, Any}
    irep :: Dict{Symbol, Any}
end

## Conversion from CFG to internal representation

const RREACT = 1
const RPROD = 2
const RK = 3
const RW = 4
const RR_ABSORB = 5

function nof_vaxes(cfg)
    cfg[:vaxes] |> keys |> collect |> length
end

function types_cnt(cfg)
    CELL_TYPES_OFFSET + nof_vaxes(cfg)
end

function prep_vaxes_pools(cfg)
    map(cfg[:vaxes] |> collect) do (k,v)
        grid = Grid(zeros(cfg[:size]...), cfg[:torus])
        if haskey(v, :init)
            grid[v[:init][1],v[:init][2]] .= v[:init][3]
        end
        grid
    end
end

function index_vaxes!(vaxes)
    for (i, k) in vaxes |> keys |> enumerate
        vaxes[k][:id] = i
    end

    vaxes
end

function vax_map(vaxes)
    Dict(k => vaxes[k][:id] for k in keys(vaxes))
end

function reactions_matrix(reactions, vaxmap, col_keys = [:k, :w, :r_absorb])
    map(reactions) do r
        vcat([map(v -> (v[1], vaxmap[v[2]]), r[:react]) |> collect,
              map(v -> (v[1], vaxmap[v[2]]), r[:prod]) |> collect,],
             map(k -> r[k], col_keys))
    end
end

function conver_zg_edge_to_idxs(edge, vaxmap)
    if edge[1] == :divide
        return tuple(edge[1:2]...,
                     (map(edge[3:end]) do r
                             (vaxmap[r[1]], r[2])
                         end)...)
    end

    edge
end

function convert_zg_to_idxs(zg, vaxmap)
    Dict([vaxmap[k],
          map(zg[k]) do es
              tuple(vaxmap[es[1]], conver_zg_edge_to_idxs(es[2:end],vaxmap)...)
          end] for k in keys(zg))
end

function convert_state_to_idxs(state, vaxmap)
    Dict((kv[1] == :cum_state ? :cum_state => vaxmap[kv[2]] : kv) for kv in state)
end

#=

vaxes = @d(
:v1 => @d(:D => 0.1,
:d => 0.0001,
:rd => 0.00001,
:init => (:, 0.1)),
:a1 => @d(:D => 0.2,
:d => 0.001,
:rd => 0.00001,
:init => (:, 0.1)),
:x1 => @d(:D => 0.01,
:d => 0.001,
:rd => 0.0001),
:kill => @d(:D => 0,
:d => 0,
:rd => 0),
:dead => @d(:D => 0,
:d => 0,
:rd => 0)
)

vaxes = index_vaxes!(vaxes)
vax_name_idx = vax_map(vaxes)

reactions = [
@d(
:react => [(1, :v1), (1, :a1)],
:prod => [(1, :x1)],
:k => 0.0001,
:w => 1.0,
:r_absorb => true,
),
@d(
:react => [(1, :kill), (1, :dead)],
:prod => [(1, :kill), (1, :dead)],
:k => 0,
:w => 1000000.0,
:r_absorb => false
)
]

rmat = reactions_matrix(reactions, vax_name_idx)

zg = @d(
:v1 => [(:v1, :prod_r, 0.001), (:v1, :adhesion, 100), (:a1, :move, 12000),
(:v1, :volume, 50, 150), (:v1, :perimeter, 2, 145), (:v1, :activity, 300, 30)],
:a1 => [(:a1, :prod_r, 0.001), (:v1, :prod_v, 1), (:a1, :prod_v, 1),
(:v1, :divide, 0.5, (:a1,1)),
(:a1, :adhesion, 100), (:a1, :volume, 50, 150), (:a1, :perimeter, 2, 145)],
:kill => [(:v1, :prod_v, 1000), (:kill, :volume, 50, 150),
(:kill, :perimeter, 2, 145), (:kill, :activity, 300, 30)],
:dead => [(:dead, :kill, 1, :apoptosis, :hard)]
)

zg = convert_zg_to_idxs(zg, vax_name_idx)

state = @d(:cum_state => :v1, :cum_state_weight => 1.0, :resting_time => 0)

state = convert_state_to_idxs(state, vax_name_idx)

=#

## Preprocessing of CPM configuration 

function adhesion_params(cfg)
    cpm = cfg[:rule_graph][:cpm]
    vaxes = cfg[:vaxes]
    if haskey(cpm, :adhesion)
        return cpm[:adhesion]
    end

    n = types_cnt(cfg)
    adh_mtx = fill(convert(Float64, get(cpm, :other_adhesion, 0.)), (n,n))
    adh_mtx[1,1] = 0.0

    zg = cfg[:rule_graph][:zg]
    for v in zg |> keys
        idx = findfirst(x -> x[2] == :adhesion, zg[v])
        if idx != nothing
            i = CELL_TYPES_OFFSET + vaxes[v][:id]
            adh_mtx[i, i] = zg[v][idx][3]
        end
    end

    for i in 2:n
        for j in 2:n
            if i != j
                adh_mtx[i,j] = min(adh_mtx[i,i], adh_mtx[j,j])
            end
        end
    end

    adh_mtx
end

function cpm_params(cfg, key)
    cpm = cfg[:rule_graph][:cpm]
    vaxes = cfg[:vaxes]
    if haskey(cpm, key)
        return cpm[key]
    end

    n = types_cnt(cfg)
    w = zeros(Float64, n) # weights
    val = zeros(Float64, n) # values

    zg = cfg[:rule_graph][:zg]
    for v in zg |> keys
        idx = findfirst(x -> x[2] == key, zg[v])
        if idx != nothing
            i = CELL_TYPES_OFFSET + vaxes[v][:id]
            w[i] = zg[v][idx][3]
            val[i] = zg[v][idx][4]
        end
    end

    (w, val)
end

volume_params(cfg) = cpm_params(cfg, :volume)
perimeter_params(cfg) = cpm_params(cfg, :perimeter)
activity_params(cfg) = cpm_params(cfg, :activity)

function chemotaxis_params(cfg, fields)
    cpm = cfg[:rule_graph][:cpm]
    vaxes = cfg[:vaxes]
    if haskey(cpm, :chemotaxis)
        return cpm[chemotaxis]
    end

    n = types_cnt(cfg)
    vals = []

    zg = cfg[:rule_graph][:zg]
    for v in zg |> keys
        idx = findfirst(x -> x[2] == :move, zg[v])
        i = CELL_TYPES_OFFSET + vaxes[v][:id]
        if idx != nothing
            id = vaxes[zg[v][idx][1]][:id]
            weights = zeros(Float64, n)
            weights[i] = zg[v][idx][3]
            push!(vals, (weights, fields[id]))
        end
    end

    vals
end

function barrier_params(cfg)
    n = types_cnt(cfg)

    barrier = fill(false, n)

    if haskey(cfg, :barrier)
        vaxes = cfg[:vaxes]
        id_barrier = CELL_TYPES_OFFSET + vaxes[cfg[:barrier][:vax]][:id]
        barrier[id_barrier] = true
    end

    barrier
end


## Simulation constructor

function mk_sim!(cfg, rules = Nothing)
    index_vaxes!(cfg[:vaxes])
    vaxes = prep_vaxes_pools(cfg)

    vaxmap = vax_map(cfg[:vaxes])

    internal = Dict(
        :vaxmap => vaxmap,
        :vaxes => sort(values(cfg[:vaxes]) |> collect, by = v -> v[:id]),
        :reactions => reactions_matrix(cfg[:reactions], vaxmap),
        :zg => convert_zg_to_idxs(cfg[:rule_graph][:zg], vaxmap)
    )

    params = params_cfg(
        T = get(cfg[:rule_graph][:cpm], :T, 20),
        adhesion = AdhesionParams(adhesion_params(cfg)),
        volume = VolumeParams(volume_params(cfg)...),
        perimeter = PerimeterParams(perimeter_params(cfg)...),
        chemotaxis = map(p -> ChemotaxisParams(p...), chemotaxis_params(cfg, vaxes)),
        activity = ActivityParams(Val{:geom}(), activity_params(cfg)...),
        barrier = BarrierParams(barrier_params(cfg))
    )

    rules_cfg = mk_cfg(post_mcs_listeners = [partial(rules, cfg)])
    full_cfg = merge(StatsCfg, VolumeCfg, AdhesionCfg, PerimeterCfg, ChemotaxisCfg, ActivityCfg, BarrierCfg,
                     params, rules_cfg)
    cpm = make_preinit_cpm(cfg[:size], cfg[:seed], cfg = full_cfg, is_torus = all(identity, cfg[:torus]))

    cfg[:sim] = Sim(Cell[], vaxes, cpm, cfg, internal)
    cfg[:sim]
end

function mk_barier!(cfg)
    if !haskey(cfg, :barrier)
        return
    end

    vaxes = cfg[:vaxes]
    model = cfg[:sim].model
    id_barrier = CELL_TYPES_OFFSET + vaxes[cfg[:barrier][:vax]][:id]
    id = make_new_cell_id!(model, id_barrier)
    cfg[:barrier][:id] = id
    # cfg.sim.model.grid[cfg[:barrier][:pos]] .= id
    for idx in cfg[:barrier][:pos]
        setpix!(model, idx, id)
    end
end

function mk_cell!(sim, state, pos, init_receptors)
    id = seed_cell_at!(sim.model, cum_state_to_idx(state[:cum_state]), pos)
    push!(sim.cells, Cell(id, init_receptors, state))
end

function init_receptors(receptors, vaxes)
    rs = zeros(Float64, vaxes |> keys |> length)
    for (r, v) in receptors
        rs[vaxes[r][:id]] = v
    end
    rs
end

function mk_cells!(sim)
    cfg = sim.cfg

    vaxes = cfg[:vaxes]
    for c in cfg[:cells]
        state = convert_state_to_idxs(c[:state], sim.irep[:vaxmap])
        mk_cell!(sim, state, CartesianIndex(c[:init_pos]), init_receptors(c[:receptors], vaxes))
    end
end

#=
cfg = Dict(
    :size => (100, 100),
    :torus => (true, true),
    :seed => 1234,

    :sim => nothing,

    :vaxes => @d(
        :v1 => @d(:D => 0.1,
                  :d => 0.0001,
                  :rd => 0.00001,
                  :init => (:, :, 0.1)
                 ),
        :a1 => @d(:D => 0.2,
                  :d => 0.001,
                  :rd => 0.00001,
                  :init => (:, :, 0.2)
                 ),
        :x1 => @d(:D => 0.01,
                  :d => 0.001,
                  :rd => 0.0001),
        :v2 => @d(:D => 0.1,
                  :d => 0.0001,
                  :rd => 0.00001
                 ),
        :a2 => @d(:D => 0.2,
                  :d => 0.001,
                  :rd => 0.00001,
                  :init => (:, :, 0.1)
                  ),
        :x2 => @d(:D => 0.01,
                  :d => 0.001,
                  :rd => 0.0001)
    ),

    :reactions => [
        @d(
            :react => [(1, :v1), (1, :a1)],
            :prod => [(1, :x1)],
            :k => 0.0001,
            :w => 1.0,
            :r_absorb => true,
        ),
        @d(
            :react => [(1, :v2), (1, :a2)],
            :prod => [(1, :x2)],
            :k => 0.0001,
            :w => 1.0,
            :r_absorb => true,
        )
    ],

    :cells => [
      @d(
        :receptors => @d(:v1 => 0.05, :v2 => 0.05),
        :state => @d(:cum_state => :v1, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (50,50)
      )
    ],

    :rule_graph => @d(
        :min_weight => 0.0,
        :resting_time => 1,
        :zg => @d(
            :v1 => [(:v1, :prod_r, 0.001), (:v1, :prod_v, 1), (:v1, :divide, 0.1), (:v1, :kill, 1, :apoptosis, :hard),
                    (:v1, :adhesion, 100), (:a1, :move, 11000),
                    (:v1, :volume, 50, 50), (:v1, :perimeter, 2, 80), (:v1, :activity, 300, 30)],
            :a1 => [(:a1, :prod_r, 0.001), (:v1, :prod_v, 1), (:a1, :prod_v, 1), (:a1, :adhesion, 100),
                    (:a1, :volume, 50, 50), (:a1, :perimeter, 2, 45)],
            :x1 => []
        ),
        :cpm => @d(
            :T => 20,
            :other_adhesion => 20,
        )
    )
)

sim = mk_sim!(cfg, (_,_) -> nothing)

mk_cells!(sim)

time_step!(sim.model)
=#

## Vax processes

function perform_reaction!(out_vaxes, in_vaxes, reaction, cfg)
    k = reaction[RK]
    if k == 0 return end

    r = ones(cfg[:size])

    for (n, v) in reaction[RREACT]
        @. r *= in_vaxes[v] ^ n
    end

    for (n, v) in reaction[RPROD]
        @. out_vaxes[v] += n * r * k
    end

    for (n, v) in reaction[RREACT]
        @. out_vaxes[v] -= n * r * k
    end
end

function perform_diffusion!(field, D, barrier_idxs)
    if D == 0 return end

    dst = zeros(Float64, size(field))
    M = fill(D, size(field))
    if barrier_idxs != nothing
        M[barrier_idxs] .= 0
    end
    # B = masked_laplacian(2) |> LocalFilters.Kernel
    # mask_smooth_torus!(dst, field.x, M, B)
    B = OffsetArray(masked_laplacian(2), -1:1,-1:1)
    masked_manifold_smoothen!(dst, field.x, M, B)
    field.x .+= dst
end

function perform_decay!(field, d)
    field.x .*= 1.0 - d
end

function correct_vals!(field)
    imin = first(CartesianIndices(field.x))
    imax = last(CartesianIndices(field.x))

    @inbounds @simd for i in imin:imax
        if field.x[i] < 0
            field.x[i] = 0
        end
    end
end


function absorb_vaxes!(vax, out_vaxes, receptors, reaction, fields, cells_border_pixels)
    lp = 1.0
    for (n, v) in reaction[RREACT]
        lp *= (v == vax ? receptors[v] : out_vaxes[v]) ^ n
    end

    lp *= reaction[RK]
    l = length(cells_border_pixels)

    for (n, v) in reaction[RREACT]
        if v == vax
            continue
        end

        fields[v].x[cells_border_pixels] .-= n*lp/l
    end
end

function produce_vax!(field, cellpixels, val)
    field.x[cellpixels] .+= val / length(cellpixels)
end

function produce_r_vax!(cell, edge)
    idx = edge[1]
    cell.receptors[idx] += (get(edge, 4, 0) > 0
                            ? edge[3] * get(edge, 4, 0) * cell.state[:cum_state_weight]
                            : edge[3])
    if cell.receptors[idx] < 0
        cell.receptors[idx] = 0
    end
end

function produce_v_vax!(cell, edge, sim, fields)
    vaxes = sim.cfg[:vaxes]
    cellpixels = get(get_stat(sim.model, :border_pixels_by_cell), cell.id, [])

    field = fields[edge[1]]
    vol = (get(edge, 4, 0) > 0
           ? get(edge, 4, 0) * cell.state[:cum_state_weight] * edge[3]
           : edge[3])
    produce_vax!(field, cellpixels, vol)
    for idx in cellpixels
        if field.x[idx] < 0
            field.x[idx] = 0
        end
    end
end

## Vax functionality

function reactions_for(reactions, vax)
    filter(reactions) do r
        findfirst(x -> x[2] == vax, r[RREACT]) != nothing
    end
end

function get_outer_vaxes(vaxes, cellpixels)
    map(vaxes) do vax
        vax[cellpixels] |> sum
    end
end

function get_bound_vaxes(nof_vaxes, cells, cell_border_list, cells_border_pixels)
    bound_vaxes = zeros(Float64, nof_vaxes)
    for id in (@>> cell_border_list keys collect filter(isneqv(BGID)))
        cidx = findfirst(c -> c.id == id, cells)
        if cidx != nothing
            bound_vaxes .+= cells[cidx].receptors .* (cell_border_list[id] / length(cells_border_pixels[id]))
        end
    end
    bound_vaxes
end

## Cell functionality

cum_state_to_idx(cum_state) = cum_state + CELL_TYPES_OFFSET

function cur_cum_state(sim, the_cell, in_vaxes)
    cfg = sim.cfg
    reactions = sim.irep[:reactions]
    id = the_cell.id

    cells_border_pixels = get_stat(sim.model, :border_pixels_by_cell)
    cell_border_list = get(get_stat(sim.model, :cell_neighbor_list), id, Dict())

    n = cfg[:vaxes] |> length
    out_vaxes = get_outer_vaxes(in_vaxes, get(cells_border_pixels, id, []))
    bound_vaxes = get_bound_vaxes(n, sim.cells, cell_border_list, cells_border_pixels)
    out_bound_vaxes = out_vaxes .+ bound_vaxes

    weights = zeros(Float64, n)
    for vax in 1:n
        rs = reactions_for(reactions, vax)
        wp = 0.0
        for r in rs
            if r[RW] == 0
                continue
            end

            lp = 1.0
            for (n, v) in r[RREACT]
                lp *= (v == vax ? the_cell.receptors[v] : out_bound_vaxes[v]) ^ n
            end
            wp += lp * r[RW]

            if r[RR_ABSORB]
                absorb_vaxes!(vax, out_vaxes, the_cell.receptors,
                              r, sim.vaxes, get(cells_border_pixels, id, []))
            end
        end
        weights[vax] = wp
    end

    idx = argmax(weights)

    if iszero(weights) || weights[idx] < cfg[:rule_graph][:min_weight] || weights[idx] == 0.0
        the_cell.state[:cum_state], the_cell.state[:cum_state_weight]
    else
        idx, weights[idx]
    end
end

function cell_divide!(cell, edge, sim)
    rs = cell.receptors |> sum
    max_rs = typeof(edge[3]) <: Number ? edge[3] : edge[3][1]
    max_rs_div = typeof(edge[3]) <: Number ? 0.5 : edge[3][2]

    if rs < max_rs
        return nothing
    end

    println("diving cell $(cell.id). num of receprors $rs threshold $(edge[3])")

    nid = divide_cell!(sim.model, cell.id)
    new_cell = Cell(nid, deepcopy(cell.receptors), deepcopy(cell.state))
    mult = fill(0.5, length(cell.receptors))

    for (r, m) in edge[4:end]
        mult[r] = m
    end

    cell.receptors .*= mult

    if sum(cell.receptors) >= max_rs
        cell.receptors .*= max_rs_div
    end

    @. mult = 1.0 - mult
    new_cell.receptors .*= mult

    if sum(new_cell.receptors) >= max_rs
        new_cell.receptors .*= max_rs_div
    end

    new_cell.state[:cum_state] = edge[1]

    new_cell
end

function necrosis!(cell, sim, vaxes)
    cellpixels = get(get_stat(sim.model, :pixels_by_cell), cell.id, [])

    for idx in 1:length(cell.receptors)
        produce_vax!(vaxes[idx], cellpixels, cell.receptors[idx])
    end
end

function hard_kill!(cell, sim)
    kill_cell!(sim.model, cell.id)
end

function soft_kill!(cell, sim)
    set_cell_kind!(sim.model, cell.id, CELL_TYPE_DEAD)
end

function cell_kill!(cell, edge, sim, vaxes)
    if rand() > edge[3]
        return false
    end

    println("killing cell $(cell.id) by $edge")

    @match edge[4] begin
        :apoptosis => nothing
        :necrosis => necrosis!(cell, sim, vaxes)
    end

    @match edge[5] begin
        :hard => hard_kill!(cell, sim)
        :soft => soft_kill!(cell, sim)
    end

    true
end

function update_by_zg!(sim, cell, vaxes)
    killed, new_cell = false, nothing

    for edge in sim.irep[:zg][cell.state[:cum_state]]
        action = edge[2]
        if action == :prod_r
            produce_r_vax!(cell, edge)
        elseif action == :prod_v
            produce_v_vax!(cell, edge, sim, vaxes)
        elseif action == :divide
            new_cell = cell_divide!(cell, edge, sim)
        elseif action == :kill
            killed = cell_kill!(cell, edge, sim, vaxes)
        end
    end

    res = killed ? [] : [cell]
    if new_cell != nothing
        push!(res, new_cell)
    end

    res
end

function get_resting_time(graph, zg, cstate)
    zg_actions = zg[cstate]
    rt_pos = findfirst(x -> x[2] == :resting_time, zg_actions)

    if rt_pos == nothing
        return graph[:resting_time]
    end

    zg_actions[rt_pos][3]
end

function update_cum_state(sim, cell, in_vaxes)
    if cell.state[:resting_time] > 0
        cell.state[:resting_time] -= 1
        return
    end

    cstate, cweight = cur_cum_state(sim, cell, in_vaxes)
    if cell.state[:cum_state] != cstate
        println("Update cum state for cell $(cell.id): old($(cell.state[:cum_state]), $(cell.state[:cum_state_weight])); new($(cstate), $cweight)")
    end
    cell.state[:cum_state] = cstate
    cell.state[:cum_state_weight] = cweight
    cell.state[:resting_time] = get_resting_time(sim.cfg[:rule_graph], sim.irep[:zg], cstate)
    set_cell_kind!(sim.model, cell.id, cum_state_to_idx(cstate))
end

function update_r_vaxes(sim, cell)
    vaxes = sim.irep[:vaxes]

    for (idx, vinfo) in enumerate(vaxes)
        cell.receptors[idx] *= 1 - vinfo[:rd]
    end
end

#### Rules

function rules(cfg, model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    if !get(cfg, :rules_enabled, true) return end

    sim = cfg[:sim]
    in_vaxes = deepcopy(sim.vaxes)
    barrier_idxs = haskey(cfg, :barrier) ? cfg[:barrier][:pos] : nothing

    for c in sim.cells
        update_cum_state(sim, c, in_vaxes)
    end

    cells = []
    for c in sim.cells
        new_cells = update_by_zg!(sim, c, sim.vaxes)
        update_r_vaxes(sim, c)
        cells = vcat(cells, new_cells)
    end
    sim.cells = cells

    reset_stat_state!(sim.model)

    for i in 1:get(cfg, :vaxes_rule_steps, 1)
        if i > 1
            in_vaxes = deepcopy(sim.vaxes)
        end

        for r in sim.irep[:reactions]
            perform_reaction!(sim.vaxes, in_vaxes, r, cfg)
        end

        vaxes = sim.irep[:vaxes]
        for (idx,vinfo) in enumerate(vaxes)
            field = sim.vaxes[idx]
            perform_diffusion!(field, vinfo[:D], barrier_idxs)
            perform_decay!(field, vinfo[:d])
            correct_vals!(field)
        end
    end
end

#### Visualization

rc(hex) = ((hex & 0xff0000) >> 16) / 255
gc(hex) = ((hex & 0xff00) >> 8) / 255
bc(hex) = (hex & 0xff) / 255
hex2rgba(hex) = RGBA(rc(hex), gc(hex), bc(hex), 1.)

default_colors = Dict(
    0 => hex2rgba(0x370617), # cell border
    1 => RGBA(1,1,1,1), # bg
    2 => hex2rgba(0x9fa6a6),
    3 => hex2rgba(0xf1f3c2),
    4 => hex2rgba(0xf3e700),
    5 => hex2rgba(0xf3d368),
    6 => hex2rgba(0xf3ba9e),
    7 => hex2rgba(0xf37b35),
    8 => hex2rgba(0xf31000),
    9 => hex2rgba(0xf3b4f1),
    10 => hex2rgba(0xf300ae),
    11 => hex2rgba(0x9a00f3),
    12 => hex2rgba(0x6100f3),
    13 => hex2rgba(0xb48bf3),
    14 => hex2rgba(0x648ff3),
    15 => hex2rgba(0xadd6f3),
    16 => hex2rgba(0x1119f3),
    17 => hex2rgba(0x1df3e8),
    18 => hex2rgba(0x20f386),
    19 => hex2rgba(0xbff3a3),
    20 => hex2rgba(0x00a876),
    21 => hex2rgba(0x961200),
    22 => hex2rgba(0xffeedb)
)

default_colors_field = Dict(
    1 => hex2rgba(0xf1f3c2),
    2 => hex2rgba(0xf3e700),
    3 => hex2rgba(0xf3d368),
    4 => hex2rgba(0xf3ba9e),
    5 => hex2rgba(0xf37b35),
    6 => hex2rgba(0xf31000),
    7 => hex2rgba(0xf3b4f1),
    8 => hex2rgba(0xf300ae),
    9 => hex2rgba(0x9a00f3),
    10 => hex2rgba(0x6100f3),
    11 => hex2rgba(0xb48bf3),
    12 => hex2rgba(0x648ff3),
    13 => hex2rgba(0xadd6f3),
    14 => hex2rgba(0x1119f3),
    15 => hex2rgba(0x1df3e8),
    16 => hex2rgba(0x20f386),
    17 => hex2rgba(0xbff3a3),
    18 => hex2rgba(0x00a876),
    19 => hex2rgba(0x961200),
    20 => hex2rgba(0xffeedb)
)

function draw_cpm(sim;
                  cell_kind_color = default_colors,
                  num_cell_kinds = nothing,
                  show_activity = false,
                  show_fields = false,
                  field_color = default_colors_field,
                  vox = nothing)
    cpm = sim.model
    if show_fields
        for v in values(sim.cfg[:vaxes])
            if get(v,:show,false)
                # vox = draw_field_contour(sim.vaxes[v[:id]],
                #                          haskey(v,:color) ? v[:color] : field_color[v[:id]],
                #                          vox = vox, nsteps = 30)
                vox = draw_field(sim.vaxes[v[:id]],
                                 haskey(v,:color) ? v[:color] : field_color[v[:id]],
                                 vox = vox)
            end
        end
    end

    vox = draw_cells(cpm, cid -> cell_kind_color[cell_kind(cpm, cid)], vox = vox)

    num_cell_kinds = num_cell_kinds == nothing ? maximum(cpm.state.cell_to_kind) : num_cell_kinds

    for k in 2:num_cell_kinds
        draw_cell_borders(cpm, k, cell_kind_color[0], vox = vox)
        if show_activity
            draw_activity_values(cpm, k, vox = vox)
        end
    end

    vox
end

### Simulation control

function burnin!(cpm, steps)
    for i in 1:steps
        time_step!(cpm)
    end
end

function prep_vis_sim(sim;
                      burnin_steps = 20,
                      num_cell_kinds = nothing,
                      show_activity = false,
                      show_fields = false,
                      vox = nothing,
                      vox_rect = nothing,
                      cell_kind_color = default_colors,
                      field_color = default_colors_field)
    burnin!(sim.model, burnin_steps)

    # t = Node(0.)
    t = Observable(0.0)

    fig = lift(t) do _
        time_step!(sim.model)
        draw_cpm(sim, num_cell_kinds = num_cell_kinds,
                 show_activity = show_activity,
                 show_fields = show_fields,
                 cell_kind_color = cell_kind_color,
                 field_color = field_color,
                 vox = vox_rect == nothing ? vox : view(vox, vox_rect...))
    end

    image(fig), t
end

function anim_sim(cpm, t; num_of_steps = 1000, fps = 1.0/30.)
    for i in 1:num_of_steps
        t[] = i
        sleep(fps)
    end
end

function record_sim(img; filename = "simulation.mp4", timestamps = 0.0:1.0/30.:33.0)
    record(img, filename, timestamps; framerate = 1.0/timestamps.step.hi) do tt
        t[] = tt
    end
end

const DEFAULT_RUNTIME_PARAMS = @d(
    :burnin_steps => 20,
    :show_fields => true,
    :show_activity => true,
    :show_sim => true,
    :cell_kind_color => default_colors,
    :field_color => default_colors_field
)

function init_sim(cfg)
    sim = mk_sim!(cfg, rules)
    mk_barier!(cfg)
    mk_cells!(sim)

    rules_enabled = get(cfg, :rules_enabled, true)

    img, t = nothing, nothing

    cfg[:runtime] = merge(DEFAULT_RUNTIME_PARAMS, get(cfg,:runtime,@d()))

    cfg[:rules_enabled] = false
    if cfg[:runtime][:show_sim]
        img,t = prep_vis_sim(sim,
                             burnin_steps = cfg[:runtime][:burnin_steps],
                             num_cell_kinds = CELL_TYPES_OFFSET+length(sim.vaxes),
                             show_fields = cfg[:runtime][:show_fields],
                             show_activity = cfg[:runtime][:show_activity],
                             cell_kind_color = cfg[:runtime][:cell_kind_color],
                             field_color = cfg[:runtime][:field_color])
        display(img)
    else
        burnin!(sim.model, cfg[:runtime][:burnin_steps])
    end
    cfg[:rules_enabled] = rules_enabled

    @d(
        :sim => sim,
        :img => img,
        :t => t
    )
end

function simulate(sim_desc; num_of_steps = 1, record = false, filename = "simulation.mp4", fps = 1.0/30.)
    if sim_desc[:sim].cfg[:runtime][:show_sim]
        if record
            record_sim(sim_desc[:img], filename = filename, timestamps = 0.:fps:num_of_steps*fps)
        else
            anim_sim(sim_desc[:sim].model, sim_desc[:t], num_of_steps = num_of_steps, fps = fps)
        end
    else
        burnin!(sim_desc[:sim].model, num_of_steps)
    end
end

#=

# Wall

cfg = Dict(
    :size => (100, 100),
    :torus => (true, true),
    :seed => 1234,

    :sim => nothing,

    :vaxes_rule_steps => 10,

    :barrier => @d(
        :vax => :wall,
        :pos => vcat(
            [CartesianIndex(i, 5) for i in 5:95],
            [CartesianIndex(5, i) for i in 5:25],
            [CartesianIndex(i, 25) for i in 5:75],
            [CartesianIndex(75, i) for i in 25:95],
            [CartesianIndex(95, i) for i in 5:95],
            [CartesianIndex(i, 95) for i in 75:95]
        )
    ),

    :vaxes => @d(
        :wall => @d(:D => 0,
                    :d => 0,
                    :rd => 0),
        :e => @d(:D => 0.02,
                 :d => 0.0,
                 :rd => 0,
                 :show => true
                 ),
        :re => @d(:D => 0,
                  :d => 0,
                  :rd => 0),
    ),

    :reactions => [
        @d(
            :react => [(1, :re), (1, :e)],
            :prod => [(1, :re)],
            :k => 0.001,
            :w => 1.0,
            :r_absorb => true,
        )
    ],

    :cells => [
        @d(
            :receptors => @d(:re => 1.0),
            :state => @d(:cum_state => :re, :cum_state_weight => 1.0, :resting_time => 0),
            :init_pos => (10,10)
        )
    ],

    :rule_graph => @d(
        :min_weight => 0.0,
        :resting_time => 1,
        :zg => @d(
            :wall => [],
            :e => [],
            :re => [(:re, :adhesion, 100), (:re, :volume, 50, 30),
                    (:re, :perimeter, 2, 70),
                    (:e, :move, 12000), (:re, :activity, 200, 20)]
        ),
        :cpm => @d(
            :T => 20,
            :other_adhesion => 20,
        )
    ),

    :runtime => @d()
)

sim_desc = init_sim(cfg)

sim_desc[:sim].vaxes[2][6:94,6:24] .= 0.1
sim_desc[:sim].vaxes[2][26:94,6:24] .= 0.3
sim_desc[:sim].vaxes[2][46:94,6:24] .= 0.7
sim_desc[:sim].vaxes[2][76:94,6:94] .= 1.0
sim_desc[:sim].vaxes[2][76:94,26:94] .= 2.0
sim_desc[:sim].vaxes[2][76:94,46:94] .= 3.0
sim_desc[:sim].vaxes[2][76:94,76:94] .= 4.0

simulate(sim_desc, num_of_steps = 100)

=#


#=
# B cell
cfg = Dict(
    :size => (100, 100),
    :torus => (true, true),
    :seed => 1234,

    :sim => nothing,

    :vaxes_rule_steps => 10,

    :vaxes => @d(
        :t => @d(:D => 0.01,
                  :d => 0.0001,
                  :rd => 0.00001,
                  :show => true,
                  :init => (:, :, 0.1)
                 ),
        :e => @d(:D => 0.02,
                  :d => 0.001,
                  :rd => 0.00001,
                  :show => true,
                 ),
        :te => @d(:D => 0,
                  :d => 0.001,
                  :rd => 0.0001)
    ),

    :reactions => [
        @d(
            :react => [(1, :t), (1, :e)],
            :prod => [(1, :te)],
            :k => 0.001,
            :w => 1.0,
            :r_absorb => true,
        )
    ],

    :cells => [
      @d(
        :receptors => @d(:e => 0.05),
        :state => @d(:cum_state => :e, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (50,50)
      )
    ],

    :rule_graph => @d(
        :min_weight => 0.0,
        :resting_time => 1,
        :zg => @d(
            :e => [(:e, :prod_r, 0.001), (:e, :prod_v, 1),
                   (:e, :adhesion, 100), (:e, :volume, 50, 50), (:e, :perimeter, 2, 85),
                   (:t, :move, 12000), (:e, :activity, 200, 30)],
            :t => [(:t, :adhesion, 100), (:t, :volume, 50, 50), (:t, :perimeter, 2, 45)],
            :te => []
        ),
        :cpm => @d(
            :T => 20,
            :other_adhesion => 20,
        )
    ),

    :runtime => @d()
)

sim_desc = init_sim(cfg)

simulate(sim_desc, num_of_steps = 100)

=#

#=

Switch between two vaxes

cfg = Dict(
    :size => (100, 100),
    :torus => (true, true),
    :seed => 1234,

    :sim => nothing,

    :vaxes_rule_steps => 10,

    :vaxes => @d(
        :t => @d(:D => 0.01,
                  :d => 0.00001,
                  :rd => 0.00001,
                  :show => true,
                  :init => (1:10, 1:10, 1)
                 ),
        :tt => @d(:D => 0.05,
                  :d => 0.00001,
                  :rd => 0.00001,
                  :show => true,
                  :init => (60:69, 60:69, 2)
        ),
        :e => @d(:D => 0,
                  :d => 0.001,
                  :rd => 0.00001,
                 ),
        :ee => @d(:D => 0,
                  :d => 0.001,
                  :rd => 0.00001,
        ),
        :te => @d(:D => 0,
                  :d => 0.001,
                  :rd => 0.0001)
    ),

    :reactions => [
        @d(
            :react => [(1, :t), (1, :e)],
            :prod => [(1, :te)],
            :k => 0.001,
            :w => 1.0,
            :r_absorb => true,
        ),
        @d(
            :react => [(1, :t), (1, :ee)],
            :prod => [(1, :te)],
            :k => 0.001,
            :w => 1.0,
            :r_absorb => true,
        )
    ],

    :cells => [
      @d(
        :receptors => @d(:e => 0.1),
        :state => @d(:cum_state => :e, :cum_state_weight => 1.0, :resting_time => 0),
        :init_pos => (50,50)
      )
    ],

    :rule_graph => @d(
        :min_weight => 0.0,
        :resting_time => 1,
        :zg => @d(
            :e => [(:ee, :prod_r, 0.0001),
                   (:e, :adhesion, 100), (:e, :volume, 50, 50), (:e, :perimeter, 2, 85),
                   (:t, :move, 12000), (:e, :activity, 200, 30)],
            :ee => [(:ee, :prod_r, 0.001),
                    (:ee, :adhesion, 100), (:ee, :volume, 50, 50), (:ee, :perimeter, 2, 85),
                    (:tt, :move, 15000), (:ee, :activity, 200, 50)],
            :t => [(:t, :adhesion, 100), (:t, :volume, 50, 50), (:t, :perimeter, 2, 45)],
            :tt => [(:tt, :adhesion, 100), (:tt, :volume, 50, 50), (:tt, :perimeter, 2, 45)],
            :te => []
        ),
        :cpm => @d(
            :T => 20,
            :other_adhesion => 20,
        )
    ),

    :runtime => @d()
)

sim_desc = init_sim(cfg)

simulate(sim_desc, num_of_steps = 100)


=#

#=

Switch between two vaxes and cells

cfg = Dict(
    :size => (100, 100),
    :torus => (true, true),
    :seed => 1234,

    :sim => nothing,

    :vaxes_rule_steps => 10,

    :vaxes => @d(
        :t => @d(:D => 0.01,
                 :d => 0.00001,
                 :rd => 0.00001,
                 :show => true,
                 :init => (1:10, 1:10, 1)
        ),
        :tt => @d(:D => 0.05,
                  :d => 0.000001,
                  :rd => 0.00001,
                  :show => true,
                  :init => (60:69, 60:69, 2)
        ),
        :e => @d(:D => 0,
                  :d => 0.001,
                  :rd => 0.000001,
                 ),
        :e0 => @d(:D => 0,
                  :d => 0.00001,
                  :rd => 0.000001,
        ),
        :ee => @d(:D => 0,
                  :d => 0.001,
                  :rd => 0.00001,
        ),
        :eee => @d(:D => 0,
                   :d => 0.001,
                   :rd => 0.00001,
        ),
        :s1 => @d(:D => 0,
                 :d => 0.001,
                 :rd => 0.00001,
        ),
        :s2 => @d(:D => 0,
                  :d => 0.001,
                  :rd => 0.00001,
        ),
        :te => @d(:D => 0,
                  :d => 0.001,
                  :rd => 0.0001)
    ),

    :reactions => [
        @d(
            :react => [(1, :t), (1, :e)],
            :prod => [(1, :te)],
            :k => 0.001,
            :w => 1.0,
            :r_absorb => true,
        ),
       @d(
           :react => [(1, :t), (1, :e0)],
           :prod => [(1, :te)],
           :k => 0.001,
           :w => 1.0,
           :r_absorb => true,
        ),
        @d(
            :react => [(1, :t), (1, :eee)],
            :prod => [(1, :te)],
            :k => 0.001,
            :w => 1.0,
            :r_absorb => true,
        ),
        @d(
            :react => [(1, :tt), (1, :ee)],
            :prod => [(1, :te)],
            :k => 0.001,
            :w => 1.0, # 10000000.0,
            :r_absorb => true,
        ),
        @d(
           :react => [(1, :t), (1, :ee)],
           :prod => [(1, :te)],
           :k => 0.001,
           :w => 1.0,
           :r_absorb => true,
        ),
        @d(
            :react => [(1, :s1), (1, :s2)],
            :prod => [(1, :te)],
            :k => 0.001,
            :w => 10.0,
            :r_absorb => false,
       )
    ],

    :cells => [
      @d(
        :receptors => @d(:e0 => 1),
        :state => @d(:cum_state => :e0, :cum_state_weight => 1.0, :resting_time => 300),
        :init_pos => (50,50)
      ),
      @d(
          :receptors => @d(:s2 => 1),
          :state => @d(:cum_state => :s1, :cum_state_weight => 1.0, :resting_time => 0),
          :init_pos => (65,65)
      )
    ],

    :rule_graph => @d(
        :min_weight => 0.0,
        :resting_time => 1,
        :zg => @d(
            :e0 => [(:e, :prod_r, 0.001),
                    (:e0, :adhesion, 100), (:e0, :volume, 50, 50), (:e0, :perimeter, 2, 85),
                    (:t, :move, 12000), (:e0, :activity, 200, 30)],
            :e => [(:ee, :prod_r, 0.01),
                   (:e, :adhesion, 100), (:e, :volume, 50, 50), (:e, :perimeter, 2, 85),
                   (:t, :move, 12000), (:e, :activity, 200, 30)],
            :ee => [(:ee, :prod_r, 0.001), (:s1, :prod_r, 0.02),
                    (:ee, :adhesion, 100), (:ee, :volume, 50, 50), (:ee, :perimeter, 2, 85),
                    (:tt, :move, 15000), (:ee, :activity, 200, 50)],
            :s1 => [(:tt, :prod_v, 0.02),
                    (:s1, :adhesion, 100), (:s1, :volume, 50, 50), (:s1, :perimeter, 2, 85),
                    (:tt, :move, 12000), (:s1, :activity, 200, 30)],
            :s2 => [(:eee, :prod_r, 0.0001), (:eee, :prod_v, 0.0001),
                    (:s2, :adhesion, 100), (:s2, :volume, 50, 50), (:s2, :perimeter, 2, 85),
                    (:t, :move, 15000), (:s2, :activity, 200, 30)],
            :eee => [(:eee, :prod_r, 0.0001), (:eee, :prod_v, 0.0001),
                     (:eee, :adhesion, 100), (:eee, :volume, 50, 50), (:eee, :perimeter, 2, 85),
                     (:t, :move, 12000), (:eee, :activity, 200, 30)],
            :t => [(:t, :adhesion, 100), (:t, :volume, 50, 50), (:t, :perimeter, 2, 45)],
            :tt => [(:tt, :adhesion, 100), (:tt, :volume, 50, 50), (:tt, :perimeter, 2, 45)],
            :te => [(:te, :adhesion, 100), (:te, :volume, 50, 50), (:te, :perimeter, 2, 45)]
        ),
        :cpm => @d(
            :T => 20,
            :other_adhesion => 20,
        )
    ),

    :runtime => @d()
)

sim_desc = init_sim(cfg)

simulate(sim_desc, num_of_steps = 100)


=#



#=
Test

cfg = Dict(
    :size => (200, 200),
    :torus => (true, true),
    :seed => rand(Int) |> abs, #1234,

    :sim => nothing,

    :vaxes => @d(
        :pat => @d(:D => 0.1,    # diffusion coefficient in environment - cim mensi, tim hur difunduje?
                  :d => 0.00001, #0.00001 # decay ratio in environment - cim vetsi, tim rychleji se rozpada?
                  :rd => 0.1,    # decay ratio on membrane
                  :show => true,
                  :init => (:,:, 0.1)), # init concentration in environment
        :ab => @d(:D => 0.21,
                  :d => 0.0001, #0.0001
                  :rd => 0.00001,
                  :show => true),
        :kom => @d(:D => 0,
                   :d => 0.01,
                   :rd => 0)
    ),

    :reactions => [
        @d(
            :react => [(1, :pat), (1, :ab)],
            :prod => [(1, :kom)],
            :k => 0.05, #0.001 #0.01
            :w => 1.0,
            :r_absorb => true,
        )
    ],

    :cells => [
        @d(
            :receptors => @d(:ab => 0.05), #0.1
            :state => @d(:cum_state => :ab, :cum_state_weight => 1.0, :resting_time => 0),
            :init_pos => (100,100)
        )
    ],

    :rule_graph => @d(
        :min_weight => 0.0,
        :resting_time => 1,
        :zg => @d(
            :pat => [(:ab, :adhesion, 100), (:ab, :volume, 50, 50), (:ab, :perimeter, 2, 45)],
            :ab => [(:ab, :prod_r, 0.001), (:ab, :prod_v, 10), (:ab, :adhesion, 100), (:ab, :volume, 50, 150),
                     (:ab, :perimeter, 2, 145), (:pat, :divide, 0.5, (:ab, 1.0)),
                     (:pat, :move, 12000), (:ab, :activity, 200, 30)],
            :kom => [(:ab, :adhesion, 100), (:ab, :volume, 50, 150), (:ab, :perimeter, 2, 145)]
            ),

        :cpm => @d(
            :T => 20,
            :other_adhesion => 20,
        )
    )
)


sim_desc = init_sim(cfg)

simulate(sim_desc, num_of_steps = 1000)

=#

#=
https://ocw.mit.edu/courses/10-626-electrochemical-energy-systems-spring-2014/34aaca3a97887695dd295db7cc0fa3c0_MIT10_626S14_S11lec24.pdf
https://www.math.uci.edu/~chenlong/226/FDM.pdf
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010895


function idx2lin(idx, s)
    j = 0
    for i in length(idx.I):2
        j += (j + idx.I[i]-1)*s[i]
    end
    j + idx.I[1]
end

function idxcrop(idx, s)
    CartesianIndex(map(((i, m)) -> mod((i-1), m)+1, idx.I, s))
end

function ker_mtx(m, kernel)
    s = size(m)
    sa = prod(s)
    ks = length(kernel)
    Is = zeros(sa*ks)
    Js = zeros(sa*ks)
    Vs = zeros(sa*ks)

    for (i, idx1) in enumerate(CartesianIndices(m))
        print("$i, $idx1,\n")
        for (j, (idx2,value)) in enumerate(kernel)
            print("$j, $idx1, $(idx2lin(idx1,s)), $(idxcrop(idx1+idx2,s)), $(idx2lin(idxcrop(idx1+idx2,s),s))\n")
            Is[(i-1)*ks + j] = idx2lin(idx1,s)
            Js[(i-1)*ks + j] = idx2lin(idxcrop(idx1+idx2,s),s)
            Vs[(i-1)*ks + j] = value
        end
    end

    SparseArrays.sparse(Is,Js,Vs, sa, sa)
end


=#

function np_eq(c_new :: AbstractArray{T,N}, c :: AbstractArray{T,N},
               fi :: AbstractArray{T,N}, M :: AbstractArray{T,N},
               kernel :: AbstractArray{K,KN}, beta) where {T, N, K,KN}
    imin = first(CartesianIndices(c))
    imax = last(CartesianIndices(c))

    kmin = minimum(kernel)
    kmax = maximum(kernel)

    @inbounds for i in CartesianIndices(c_new)
        w = 0.0
        j_first = i + kmin
        j_last = i + kmax
        not_border = max(imin, j_first) == j_first && min(imax, j_last) == j_last

        if not_border
            @simd for j in kernel
                jj = i + j
                w += beta*M[jj]*(c[jj]+c[i])*(fi[jj]-fi[i])
            end
        else
            @simd for j in kernel
                jj = idxcrop(i + j, imax)
                w += beta*M[jj]*(c[jj]+c[i])*(fi[jj]-fi[i])
            end
        end

        c_new[i] = M[i]*w
    end
end


#=

m = [1 2 3 4 5 6; 1 2 3 4 5 6; 1 2 3 4 5 6; 1 2 3 4 5 6; 1 2 3 4 5 6; 1 2 3 4 5 6]
m = [0. 1. 0.; 1. -4. 1.; 0. 1. 0.]

mid = 2
m = zeros(2*mid-1,2*mid-1)
m[mid,mid] = 1.

ker = [(CartesianIndex(0,0), -4),(CartesianIndex(-1,0), 1), (CartesianIndex(1,0), 1), (CartesianIndex(0,-1), 1), (CartesianIndex(0,1), 1)]

km = ker_mtx(m, ker)

prob = LinearProblem(km, reshape(m,m |> size |> prod)) # reshape([0.0 1.0 0.0; 1.0 -4.0 1.0; 0.0 1.0 0.0],9)

sol = solve(prob)

sol = solve(prob, KrylovJL_GMRES()) # IterativeSolversJL_GMRES() KrylovJL_CRAIGMR KrylovJL_GMRES

kkm = km\reshape(m,m |> size |> prod)

kkm = [0. 0. 0.; 0. 1. 0.; 0. 0. 0.]

reshape(km*reshape(kkm, kkm |>size|>prod), size(kkm)) == m

mm = reshape(km*reshape(m, m |>size|>prod), size(m))

reshape(km\reshape(mm,mm |> size |> prod),size(mm))


=#

## ZG Manipulation

function rename_zg_edge(edge, vaxmap)
    if edge[1] == :divide
        return tuple(edge[1:2]...,
                     (map(edge[3:end]) do r
                          (get(vaxmap,r[1],r[1]), r[2])
                      end)...)
    end

    edge
end

function rename_zg(vaxmap, zg)
    Dict([get(vaxmap,k,k),
          map(zg[k]) do es
              tuple(get(vaxmap,es[1],es[1]), rename_zg_edge(es[2:end],vaxmap)...)
          end] for k in keys(zg))
end

function default_zg_combinator(edges1, edges2)
    emap = Dict([
        e[1:2], e
    ] for e in edges1)

    vcat(edges1, filter(e -> get(emap, e[1:2], nothing) == nothing, edges2))
end

function merge_zgs(combinator :: Function, vax_map :: Dict{Symbol,Symbol}, zgs...)
    new_zg = mergewith(combinator, map(zg -> rename_zg(vax_map, zg), zgs)...)
end

function merge_zgs(vax_map :: Dict{Symbol,Symbol}, zgs...)
    merge_zgs(default_zg_combinator, vax_map, zgs...)
end

function merge_zgs(zgs...)
    merge_zgs(Dict{Symbol,Symbol}(), zgs...)
end

#=

rename_zg(@d(:pat => :PAT, :ab => :AB), @d(
:pat => [(:ab, :adhesion, 100), (:ab, :volume, 50, 50), (:ab, :perimeter, 2, 45)],
:ab => [(:ab, :prod_r, 0.001), (:ab, :prod_v, 10), (:ab, :adhesion, 100), (:ab, :volume, 50, 150),
(:ab, :perimeter, 2, 145), (:pat, :divide, 0.5, (:ab, 1.0)),
(:pat, :move, 12000), (:ab, :activity, 200, 30)],
:kom => [(:ab, :adhesion, 100), (:ab, :volume, 50, 150), (:ab, :perimeter, 2, 145)]
))


merge_zgs(
@d(
:pat => [(:ab, :adhesion, 100), (:ab, :volume, 50, 50), (:ab, :perimeter, 2, 45)],
:ab => [(:ab, :adhesion, 100), (:ab, :volume, 50, 150), (:ab, :perimeter, 2, 145), (:pat, :move, 12000), (:ab, :activity, 200, 30)],
:kom => [(:ab, :adhesion, 100), (:ab, :volume, 50, 150), (:ab, :perimeter, 2, 145)]
),
@d(
:ab => [(:ab, :prod_r, 0.001), (:ab, :prod_v, 10), (:pat, :divide, 0.5, (:ab, 1.0))],
)
)

merge_zgs(
@d(
:pat => [(:ab, :adhesion, 100), (:ab, :volume, 50, 50), (:ab, :perimeter, 2, 45)],
:ab => [(:ab, :adhesion, 100), (:ab, :volume, 50, 150), (:ab, :perimeter, 2, 145), (:pat, :move, 12000), (:ab, :activity, 200, 30)],
:kom => [(:ab, :adhesion, 100), (:ab, :volume, 50, 150), (:ab, :perimeter, 2, 145)]
),
@d(
:ab => [(:ab, :prod_r, 0.001), (:ab, :prod_v, 10), (:pat, :divide, 0.5, (:ab, 1.0))],
:kom => [(:ab, :adhesion, 100), (:ab, :volume, 50, 150), (:ab, :perimeter, 2, 145)]
)
)

=#

remap_pair(vax_map, (num, id)) = (num, get(vax_map, id, id))

function remap_reaction(vax_map, reaction)
    merge(reaction, Dict(
        :react => map(x -> remap_pair(vax_map, x), reaction[:react]) |> collect,
        :prod => map(x -> remap_pair(vax_map, x), reaction[:prod]) |> collect
    ))
end

function merge_reactions(vax_map, reactions...)
    rd = Dict()
    res_rs = []
    for rs in reactions
        for r in map(r->remap_reaction(vax_map, r), rs)
            if get(rd, r, false) == false
                rd[r] = true
                push!(res_rs, r)
            end
        end
    end
    res_rs
end

function remap_vaxes(vax_map, vaxes)
    Dict([get(vax_map, vax, vax), def]
         for (vax, def) in vaxes)
end

function take_first(arg1, arg2)
    arg1
end

function merge_vaxes(combinator :: Function, vax_map :: Dict{Symbol,Symbol}, vaxes...)
    vaxes = mergewith(combinator, map(vs -> remap_vaxes(vax_map, vs), vaxes)...)
end

function merge_vaxes(vax_map :: Dict{Symbol,Symbol}, vaxes...)
    merge_vaxes(take_first, vax_map, vaxes...)
end

function merge_vaxes(vaxes...)
    merge_vaxes(take_first, Dict{Symbol,Symbol}(), vaxes...)
end

#=

merge_reactions(@d(:pat => :PAT, :ab => :AB),
 [@d(:react => [(1,:pat),(1,:ab)], :prod => [(1,:x)])])

merge_reactions(@d(),
[@d(:react => [(1,:pat),(1,:ab)], :prod => [(1,:x)])],
[@d(:react => [(1,:pat),(1,:ab)], :prod => [(1,:x)])])

merge_reactions( @d(),
[@d(:react => [(1,:pat),(1,:ab)], :prod => [(1,:patab)])],
[@d(:react => [(1,:pat),(1,:ab)], :prod => [(1,:patab)])],
[@d(:react => [(1,:v),(1,:a)], :prod => [(1,:x)])])


merge_reactions( @d(:v => :pat, :a => :ab, :x => :patab),
[@d(:react => [(1,:pat),(1,:ab)], :prod => [(1,:patab)])],
[@d(:react => [(1,:pat),(1,:ab)], :prod => [(1,:patab)])],
[@d(:react => [(1,:v),(1,:a)], :prod => [(1,:x)])])

remap_vaxes(@d(
:pat => :PAT,
:ab => :AB
),@d(
:pat => @d(:D => 0.1,
:d => 0.00001,
:rd => 0.1,
:show => true,
:init => (:,:, 0.1)),
:ab => @d(:D => 0.21,
:d => 0.0001, #0.0001
:rd => 0.00001,
:show => true),
:kom => @d(:D => 0,
:d => 0.01,
:rd => 0)
))

merge_vaxes(Dict(
:pat => :PAT,
:ab => :AB
),@d(
:pat => @d(:D => 0.1,
:d => 0.00001,
:rd => 0.1,
:show => true,
:init => (:,:, 0.1)),
:ab => @d(:D => 0.21,
:d => 0.0001, #0.0001
:rd => 0.00001,
:show => true)),
@d(:kom => @d(:D => 1,
:d => 0.01,
:rd => 0)))


merge_vaxes(Dict(
:pat => :PAT,
:ab => :AB
),@d(
:pat => @d(:D => 0.1,
:d => 0.00001,
:rd => 0.1,
:show => true,
:init => (:,:, 0.1)),
:ab => @d(:D => 0.21,
:d => 0.0001, #0.0001
:rd => 0.00001,
:show => true),
:kom => @d(:D => 0,
:d => 0.01,
:rd => 0)
),
@d(:kom => @d(:D => 1,
:d => 0.01,
:rd => 0)))

=#

function merge_cfgs(actions, mergers = Dict(:zg => merge_zgs, :vaxes => merge_vaxes, :reactions => merge_reactions))
    mapping = Dict()
    cfg = Dict()

    for action in actions
        action = copy(action)
        print("action $action \n")
        for k in [:mapping, :vaxes, :reactions, :zg]
            v = get(action, k, nothing)
            print("value $k, $v \n")
            if v == nothing
                continue
            end

            delete!(action, k)

            if k == :mapping
                mapping = v
            elseif k == :zg
                cfg[:rule_graph][:zg] = mergers[k](mapping, get(cfg[:rule_graph], :zg, Dict()), v)
            else
                default = k == :vaxes ? Dict() : []
                cfg[k] = mergers[k](mapping, get(cfg, k, default), v)
            end
        end

        cfg = mergewith(merge, cfg, action)
    end

    cfg
end

#=
merge_actions = [
    @d(:size => (100, 100),
       :torus => (true, true),
       :seed => 1234,

       :sim => nothing,

       :vaxes_rule_steps => 10,
       :rule_graph => @d(
           :min_weight => 0.0,
           :resting_time => 1,
           :zg => @d(),
           :cpm => @d(
               :T => 20,
               :other_adhesion => 20,
           )
       ),
       :runtime => @d()),
    @d(:mapping => Dict(:v => :t, :a => :e, :x => :te)),
    @d(:vaxes => @d(:v => @d(:D => 0.01,
                             :d => 0.00001,
                             :rd => 0.00001,
                             :show => true,
                             :init => (1:10, 1:10, 1)),
                    :a => @d(:D => 0,
                             :d => 0.001,
                             :rd => 0.000001),
                    :x => @d(:D => 0,
                              :d => 0.001,
                              :rd => 0.0001)),
       :reactions => [
           @d(
               :react => [(1, :v), (1, :a)],
               :prod => [(1, :x)],
               :k => 0.001,
               :w => 1.0,
               :r_absorb => true,
           ),
       ],
       :zg => @d(
           :a => [(:a, :adhesion, 100), (:a, :volume, 50, 50), (:a, :perimeter, 2, 85),
                  (:v, :move, 12000), (:a, :activity, 200, 30)],
           :v => [(:v, :adhesion, 100), (:v, :volume, 50, 50), (:v, :perimeter, 2, 45)],
           :x => [(:x, :adhesion, 100), (:x, :volume, 50, 50), (:x, :perimeter, 2, 45)]
       )),
    @d(:mapping => Dict(:v => :t, :a => :e, :x => :te, :v1 => :e0),
       :vaxes => @d(:v1 => @d(:D => 0,
                              :d => 0.00001,
                              :rd => 0.000001)),
       :reactions => [@d(
           :react => [(1, :v), (1, :v1)],
           :prod => [(1, :x)],
           :k => 0.001,
           :w => 1.0,
           :r_absorb => true,
       )],
       :zg => @d(
           :v1 => [(:a, :prod_r, 0.001),
                   (:v1, :adhesion, 100), (:v1, :volume, 50, 50), (:v1, :perimeter, 2, 85),
                   (:v, :move, 12000), (:v1, :activity, 200, 30)]
       ))

]


cfg = merge_cfgs(merge_actions)
=#

function print_zg(zg; filter_out = Set())
    out = """digraph zg {
fontname="Helvetica,Arial,sans-serif"
node [fontname="Helvetica,Arial,sans-serif"]
edge [fontname="Helvetica,Arial,sans-serif"]
rankdir=LR;
    """

    for (v,es) in zg
        for e in es
            if e[2] in filter_out
                continue
            end
            out = out * """   $(v) -> $(e[1]) [label="$(e[2:end])"];\n"""
        end
    end

    out * "}"
end

function print_reactions(reactions)
    out = """digraph reactions {
fontname="Helvetica,Arial,sans-serif"
node [fontname="Helvetica,Arial,sans-serif"]
edge [fontname="Helvetica,Arial,sans-serif"]
rankdir=LR;
    """

    rid = 1
    for r in reactions
        out = out * """   r$rid [shape="box"];\n"""
        for (n, v) in r[:react]
            out = out * """   $(v) -> r$rid [label="$(n)"];\n"""
        end
        for (n, v) in r[:prod]
            out = out * """   r$rid -> $(v)  [label="$(n)"];\n"""
        end

        rid += 1
    end

    out * "}"
end

#=

print(print_zg(cfg[:rule_graph][:zg]))

print(print_zg(cfg[:rule_graph][:zg], filter_out = Set([:perimeter, :activity, :volume, :adhesion])))

print(print_reactions(cfg[:reactions]))

echo 'digraph { a -> b }' | dot -Tsvg > output.svg



=#

### Generative simulations

function gen_vax(params)
    name = Symbol("vax_" * randstring(params[:vax_name_length]))
    Dict(
        name => Dict(:D => rand(params[:vax_D]),
                     :d => rand(params[:vax_d]),
                     :rd => rand(params[:vax_rd])))
end

function gen_vaxes(params)
    merge([gen_vax(params) for _ in 1:rand(params[:nof_vaxes])]...)
end


#=

gen_params = Dict(
    :nof_vaxes => 1:5,
    :nof_reactions => 2:5,
    :vax_name_length => 2,
    :vax_D => 0:1e-6:0.1,
    :vax_d => 0:1e-6:0.1,
    :vax_rd=> 0:1e-6:0.1,
    :reaction_stochiometry_coef => 1:1,
    :reactions_nof_reactants => 1:3,
    :reactions_nof_products => 1:1,
    :reaction_weight => 1:2,
    :reactions_rate => 0:1e-6:0.1,
    :reaction_absorb => [true, false],
    :zg_nof_edges => 0:10,
    :zg_edge_action_to_type => Dict(
        :adhesion => :physics_prop,
        :volume => :physics_prop,
        :perimeter => :physics_prop,
        :activity => :motility,
        :move => :chemo_attraction,
        :prod_v => :produce_vax,
        :prod_r => :produce_receptor,
        :divide => :divide,
        :kill =>:kill
    ),
    :zg_edge_type => Dict(
        :physics_prop => Dict(:prob => 1.0,
                              :adhesion => 90:110,
                              :adhesion_perturb_dev => 1,
                              :volume_weight=>40:60,
                              :volume=>50:60,
                              :volume_weight_perturb_dev => 1,
                              :volume_perturb_dev => 1,
                              :perimeter_weight=>1:10,
                              :perimeter => 80:90,
                              :perimeter_weight_perturb_dev => 1,
                              :perimeter_perturb_dev => 1),
        :motility => Dict(:prob => 0.5,
                          :activity_weight=>190:210,
                          :activity=>20:40,
                          :activity_weight_perturb_dev => 1,
                          :activity_perturb_dev => 1),
        :chemo_attraction => Dict(:prob => 0.2,
                                  :chemo_attraction_weight => 10000:1000:20000,
                                  :chemo_attraction_weight_perturb_dev => 1),
        :produce_vax => Dict(:prob => 0.3,
                             :times => 3,
                             :vax_amount => 0:1e-3:1,
                             :vax_dynanics => 0:1,
                             :vax_amount_perturb_dev => 1e-4),
        :produce_receptor => Dict(:prob => 0.3,
                                  :times => 3,
                                  :vax_amount => 0:1e-3:1,
                                  :vax_dynanics => 0:1,
                                  :vax_amount_perturb_dev => 1e-4),
        :divide => Dict(:prob => 0.1,
                        :divide_nof_max_receptors => 1:0.1:2,
                        :divide_nof_max_receptors_perturb_dev => 1e-4),
        :kill => Dict(:prob => 0.1,
                      :kill_prob => 0:0.1:1,
                      :kill_prob_perturb_dev => 1e-4)),
    :vax_merge_prob => 0.8,
    :zg_merge_edges_prob => 0.5,
    :vax_mutate_prob => 0.9,
    :vax_perturb_dev => 1e-4,
    :vax_mutate_name_prob => 0.01,
    :zg_mutate_prob => 0.9,
    :zg_add_edge_prob => 0.9,
    :zg_del_edge_prob => 0.9,
    :zg_mutate_edge_prob => 0.9,
    :zg_rewire_edge_prob => 0.9
)

gen_vax(gen_params)

vaxes = gen_vaxes(gen_params)

=#

function gen_reaction(vaxes, params)
    vax_name = keys(vaxes) |> collect

    Dict(
        :react => unique([(rand(params[:reaction_stochiometry_coef]), rand(vax_name))
                          for _ in 1:rand(params[:reactions_nof_reactants])]),
        :prod => unique([(rand(params[:reaction_stochiometry_coef]), rand(vax_name))
                         for _ in 1:rand(params[:reactions_nof_products])]),
        :k => rand(params[:reactions_rate]),
        :w => rand(params[:reaction_weight]),
        :r_absorb => rand(params[:reaction_absorb]),
    )
end

function gen_reactions(vaxes, params)
    [gen_reaction(vaxes, params) for _ in 1:rand(params[:nof_reactions])]
end

#=

gen_reaction(vaxes, gen_params)

reactions = gen_reactions(vaxes, gen_params)



=#

#=
edge_type = Dict(
    :physics_prop => Dict(:prob => 1.0,
                          :adhession => 90:110,
                          :volume_weight=>40:60,
                          :volume=>50:60,
                          :perimeter_weight=>1:10,
                          :perimeter => 80:90),
    :motility => Dict(:prob => 0.5,
                      :activity_weight=>190:210,
                      :activity_weight=>20:40),
    :chemo_attraction => Dict(:prob => 0.2,
                              :chemo_attraction_weight => 10000:1000:20000),
    :produce_vax => Dict(:prob => 0.3,
                         :times => 3,
                         :vax_amount => 0:1e-3:1,
                         :vax_dynanics => 0:1),
    :produce_receptor => Dict(:prob => 0.3,
                              :times => 3,
                              :vax_amount => 0:1e-3:1,
                              :vax_dynanics => 0:1),
    :divide => Dict(:prob => 0.1,
                    :divide_nof_max_receptors => 1:0.1:2),
    :kill => Dict(:prob => 0.1,
                  :kill_prob => 0:0.1:1))

=#

function gen_edge(::Val{:physics_prop}, vaxes, vname, params)
    [(vname, :adhesion, rand(params[:adhesion])),
     (vname, :volume, rand(params[:volume_weight]), rand(params[:volume])),
     (vname, :perimeter, rand(params[:perimeter_weight]), rand(params[:perimeter]))]
end

function gen_edge(::Val{:motility}, vaxes, vname, params)
    [(vname, :activity, rand(params[:activity_weight]), rand(params[:activity]))]
end

rand_vax(vaxes) = rand(keys(vaxes)|>collect)

function gen_edge(::Val{:chemo_attraction}, vaxes, vname, params)
    [(rand_vax(vaxes), :move, rand(params[:chemo_attraction_weight]))]
end

function gen_edge(::Val{:produce_vax}, vaxes, vname, params)
    [(rand_vax(vaxes), :prod_v, rand(params[:vax_amount]), rand(params[:vax_dynanics]))]
end

function gen_edge(::Val{:produce_receptor}, vaxes, vname, params)
    [(rand_vax(vaxes), :prod_r, rand(params[:vax_amount]), rand(params[:vax_dynanics]))]
end

function gen_edge(::Val{:divide}, vaxes, vname, params)
    [(vname, :divide, rand(params[:divide_nof_max_receptors]))]
end

function gen_edge(::Val{:kill}, vaxes, vname, params)
    [(vname,  :kill, rand(params[:kill_prob]), :apoptosis, :hard)]
end

function gen_edges(vaxes, vname, params)
    Tuple{Symbol, Symbol, Any, Vararg{Any}}[e
     for et in keys(params[:zg_edge_type])
         for _ in 1:get(params[:zg_edge_type][et], :times, 1)
             if params[:zg_edge_type][et][:prob] > rand()
                 for e in gen_edge(Val{et}(), vaxes, vname, params[:zg_edge_type][et])]
end


function gen_zg(vaxes, params)
    vax_name = keys(vaxes) |> collect

    Dict(vname => gen_edges(vaxes, vname, params) for vname in vax_name)
end

#=

zg = gen_zg(vaxes, gen_params)

=#

wmid(x,y,d = 0.5) = (x+y)*d

function vax_merge(vaxes1, vaxes2, params)
    if rand() > 0.5
        vaxes1, vaxes2 = vaxes2, vaxes1
    end

    vax_map = Dict{Symbol, Symbol}()
    vaxes = deepcopy(vaxes2)

    for v in keys(vaxes1)
        if rand() < params[:vax_merge_prob]
            idx = vax_map[v] = rand_vax(vaxes2)
            for k in keys(vaxes[idx])
                vaxes[idx][k] = wmid(vaxes[idx][k], vaxes1[v][k], rand())
            end
        else
            vaxes[v] = deepcopy(vaxes1[v])
        end
    end

    Dict(
        :vax_map => vax_map,
        :vaxes => vaxes
    )
end

#=

vs1 = gen_vaxes(gen_params)
vs2 = gen_vaxes(gen_params)

vs = vax_merge(vs1, vs2, gen_params)

=#

function merge_edges(e1, e2, params)
    tuple([typeof(e1[i]) <: Number ? wmid(e1[i],e2[i], rand()) : rand([e1[i], e2[i]])
           for i in 1:length(e1)]...)
end

function zg_edge_combinator(params, edges1, edges2)
    emap = Dict{Any,Any}([
        e[1:2], e
    ] for e in edges1)

    for e in edges2
        k = e[1:2]
        if get(emap, k, nothing) == nothing
            emap[k] = e
            continue
        end

        if params[:zg_merge_edges_prob] > rand()
            emap[k] = merge_edges(emap[k], e, params)
        else
            emap[k] = rand([emap[k],e])
        end
    end

    values(emap) |> collect
end

#=

merge_edges((:a, :b, 1, :c), (:a, :b, 2, :d), Dict())

=#

function zg_combine(zg1, zg2, vax_map, params)
    merge_zgs((x,y)->zg_edge_combinator(params, x, y), vax_map, zg1, zg2)
end

#=

zg1 = gen_zg(vs1, gen_params)
zg2 = gen_zg(vs2, gen_params)

zg = zg_combine(zg1, zg2, vs[:vax_map], gen_params)

=#

zero_clamp(x) = x < 0 ? 0.0 : x

function mutate_vax(vaxes, params)
    if params[:vax_mutate_prob] < rand()
        return Dict(:vaxes => vaxes, :vax_map => Dict())
    end

    vax_to_mutate = rand_vax(vaxes)
    param = rand(keys(vaxes[vax_to_mutate]))
    perturb = rand(Normal(0, params[:vax_perturb_dev]))
    new_vaxes = deepcopy(vaxes)
    new_vaxes[vax_to_mutate][param] = zero_clamp(new_vaxes[vax_to_mutate][param] + perturb)

    vax_map = Dict()
    if rand() < params[:vax_mutate_name_prob]
        new_name = Symbol("vax_" * randstring(params[:vax_name_length]))
        new_vaxes[new_name] = new_name[vax_to_mutate]
        delete!(new_name, vax_to_mutate)
        vax_map[vax_to_mutate] = new_name
    end

    Dict(:vaxes => new_vaxes, :vax_map => vax_map)
end

#=

mut_vaxes = mutate_vax(vaxes, gen_params)

=#

function mutate_edge(::Val{:adhesion}, states, edge, vname, params)
    (vname, :adhesion, zero_clamp(edge[3] + rand(Normal(0, params[:adhesion_perturb_dev]))))
end

function mutate_edge(::Val{:volume}, states, edge, vname, params)
    (vname, :volume,
     zero_clamp(edge[3] + rand(Normal(0, params[:volume_weight_perturb_dev]))),
     zero_clamp(edge[4] + rand(Normal(0, params[:volume_perturb_dev]))))
end

function mutate_edge(::Val{:perimeter}, states, edge, vname, params)
    (vname, :perimeter,
     zero_clamp(edge[3] + rand(Normal(0, params[:perimeter_weight_perturb_dev]))),
     zero_clamp(edge[4] + rand(Normal(0, params[:perimeter_perturb_dev]))))
end

function mutate_edge(::Val{:activity}, states, edge, vname, params)
    (vname, :activity,
     zero_clamp(edge[3] + rand(Normal(0, params[:activity_weight_perturb_dev]))),
     zero_clamp(edge[4] + rand(Normal(0, params[:activity_perturb_dev]))))
end

function mutate_edge(::Val{:move}, states, edge, vname, params)
    (rand_vax(vaxes), :move, zero_clamp(edge[3] + rand(Normal(0, params[:chemo_attraction_weight_perturb_dev]))))
end

function mutate_edge(::Val{:prod_v}, states, edge, vname, params)
    (rand_vax(vaxes), :prod_v,
     zero_clamp(edge[3] + rand(Normal(0, params[:vax_amount_perturb_dev]))),
     rand(params[:vax_dynanics]))
end

function mutate_edge(::Val{:prod_r}, states, edge, vname, params)
    (rand_vax(vaxes), :prod_r,
     zero_clamp(edge[3] + rand(Normal(0, params[:vax_amount_perturb_dev]))),
     rand(params[:vax_dynanics]))
end

function mutate_edge(::Val{:divide}, states, edge, vname, params)
    (vname, :divide,
     zero_clamp(edge[3] + rand(Normal(0, params[:divide_nof_max_receptors_perturb_dev]))))
end

function mutate_edge(::Val{:kill}, states, edge, vname, params)
    (vname, :kill,
     zero_clamp(edge[3] + rand(Normal(0, params[:kill_prob_perturb_dev]))),
     :apoptosis, :hard)
end

function mutate_zg(zg, params)
    if params[:zg_mutate_prob] < rand()
        return zg
    end

    new_zg = deepcopy(zg)
    src_to_mutate = rand(keys(zg))

    print("src_to_mutate: $src_to_mutate\n")

    if params[:zg_add_edge_prob] > rand()
        et = rand(keys(params[:zg_edge_type]))
        append!(new_zg[src_to_mutate], gen_edge(Val{et}(), zg, src_to_mutate, params[:zg_edge_type][et]))
    end

    if params[:zg_del_edge_prob] > rand()
        deleteat!(new_zg[src_to_mutate], rand(1:length(new_zg[src_to_mutate])))
    end

    if params[:zg_mutate_edge_prob] > rand()
        idx = rand(1:length(new_zg[src_to_mutate]))
        et = new_zg[src_to_mutate][idx][2]
        edge = new_zg[src_to_mutate][idx]
        new_zg[src_to_mutate][idx] = mutate_edge(Val{et}(),
                                                 params[:zg_rewire_edge_prob] > rand() ? keys(zg) : [edge[1]],
                                                 edge, src_to_mutate,
                                                 params[:zg_edge_type][params[:zg_edge_action_to_type][et]])
    end

    new_zg
end

#=

mutated_zg = mutate_zg(zg, gen_params)

=#
