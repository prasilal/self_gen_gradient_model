# RF CPM utilities

include("rf_cpm/DiceSets.jl")
include("rf_cpm/Grids.jl")
include("rf_cpm/GridModels.jl")
include("rf_cpm/Constraints.jl")
include("rf_cpm/Stats.jl")

struct CPMShape{TUID, T, N, Tdc, Trnd} <: AbstractAgentShape
    id :: TUID
    model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}
    attrs :: Dict{Symbol, Any}
end

get_pos(shape :: CPMShape{TUID, T, N, Tdc, Trnd}) where {TUID, T, N, Tdc, Trnd} =
    get_stat(shape.model, :centroid)[shape.id]
get_bounding_sphere(shape :: CPMShape{TUID, T, N, Tdc, Trnd}) where {TUID, T, N, Tdc, Trnd} =
    get_stat(shape.model, :bounding_ball_by_cell)[shape.id]
shape_centered(shape :: CPMShape{TUID, T, N, Tdc, Trnd}) where {TUID, T, N, Tdc, Trnd} =
    get_stat(shape.model, :pixels_by_cell)[shape.id]
neighborhood_centered(shape :: CPMShape{TUID, T, N, Tdc, Trnd}) where {TUID, T, N, Tdc, Trnd} =
    get_stat(shape.model, :border_pixels_by_cell)[shape.id]

@def_fn_rule rule_agent_cpm_iterate begin
    is_all_alloc = is_all_alloc_q(state_desc)

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0
            append_allocated(is_all_alloc, allocated, produced, used, i, j)
            foreach(allocated[i][j]) do agent
                time_step!(agent.state)
            end
        end
    end
end
