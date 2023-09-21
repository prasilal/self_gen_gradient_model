#module Stats

include("stats/border_pixels_by_cell.jl")
include("stats/bounding_ball.jl")
include("stats/cell_neighbor_list.jl")
include("stats/centroids.jl")
include("stats/connected_components_by_cell.jl")
include("stats/connected_components_by_cell_border.jl")
include("stats/connectedness.jl")
include("stats/pixels_by_cell.jl")

const STATS = Dict(
    :border_pixels_by_cell => border_pixels_by_cell_stat,
    :bounding_ball_by_cell => bounding_ball_stat,
    :cell_neighbor_list => cell_neighbor_list_stat,
    :centroid => centroid_stat,
    :connected_components_by_cell => connected_components_by_cell_stat,
    :connected_components_by_cell_border => connected_components_by_cell_border_stat,
    :connectedness => connectedness_stat,
    :pixels_by_cell => pixels_by_cell_stat
)

const StatsCfg = CPMCfg(
    Dict(),
    Function[],
    Function[],
    Function[],
    Function[],
    Function[],
    STATS
)

#end
