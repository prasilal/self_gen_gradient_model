#module Grids
#export AbstractGrid, Grid, midpoint, Stencil, moore_stencil, neumann_stencil, neigh

abstract type AbstractGrid{T, N} <: AbstractArray{T,N} end

struct Grid{T, N} <: AbstractGrid{T, N}
    x :: Array{T, N}
    torus :: NTuple{N,Bool}
end

function Grid(; sizes = (100,100), torus = false)
    if typeof(torus) == Bool
        torus = tuple(fill(torus, length(sizes))...)
    elseif torus == nothing
        torus = tuple(fill(false, length(sizes))...)
    end

    Grid{UInt,length(sizes)}(zeros(UInt, sizes...), torus)
end

@inline Base.length(g :: Grid{T, N}) where {T,N} = length(g.x)
@inline Base.checkbounds(g :: Grid{T, N}, i) where {T,N} = checkbounds(g.x, i)
@inline Base.checkbounds(g :: Grid{T, N}, idxs...) where {T,N} = checkbounds(g.x, idxs...)

@inline Base.size(g :: Grid{T, N}) where {T,N} = size(g.x)
@inline Base.getindex(g :: Grid{T, N}, i) where {T,N} = getindex(g.x, i)
@inline Base.getindex(g :: Grid{T, N}, idxs...) where {T,N} = getindex(g.x, idxs...)

@inline Base.setindex!(g :: Grid{T, N}, v :: T, i) where {T,N} = setindex!(g.x, v, i)
@inline Base.setindex!(g :: Grid{T, N}, v :: T, i...) where {T,N} = setindex!(g.x, v, i...)

@inline halve(x :: Int) = div(x, 2)
@inline midpoint(g :: Grid{T, N}) where {T,N} = halve.(size(g.x))

@inline Base.eachindex(g :: Grid{T, N}) where {T,N} = CartesianIndices(CartesianIndex(size(g.x)))

struct Stencil{N}
    x :: Array{CartesianIndex{N}, 1}
    r :: Int
    hash :: UInt
end

max_dist(xs) = reduce((m, x) -> max(x, map(abs, x) |> maximum), xs, init = 0)

function moore_stencil_(dims :: T) where {T <: Int}
    r = CartesianIndices(map(_ -> -1:1, tuple(1:dims...)))
    stencil = CartesianIndex{dims}[]
    for i in r
        if !all(iszero,i.I)
            push!(stencil, i)
        end
    end
    Stencil(stencil, 1, hash(stencil))
end

const MOORE_STENCILS = Dict(
    2 => moore_stencil_(2),
    3 => moore_stencil_(3)
)

@inline function moore_stencil(dims :: T) where {T <: Int}
    if haskey(MOORE_STENCILS, dims)
        return MOORE_STENCILS[dims]
    end
    moore_stencil_(dims)
end

function push_idx!(idxs :: Array{CartesianIndex{N}, 1}, i :: Int, v :: Int) where {N}
    idx = zeros(Int, N)
    idx[i] = v
    push!(idxs, CartesianIndex(idx...))
end

function neumann_stencil_(dims :: T) where {T <: Int}
    idxs = CartesianIndex{dims}[]
    for i in 1:dims
        push_idx!(idxs, i, -1)
        push_idx!(idxs, i, 1)
    end
    Stencil(idxs, 1, hash(idxs))
end

const NEUMANN_STENCILS = Dict(
    2 => neumann_stencil_(2),
    3 => neumann_stencil_(3)
)

@inline function neumann_stencil(dims :: T) where {T <: Int}
    if haskey(NEUMANN_STENCILS, dims)
        return NEUMANN_STENCILS[dims]
    end
    neumann_stencil_(dims)
end

@inline function is_safe_neigh(idx :: NTuple{N, Int}, r :: Int, g :: Array{T, N}) where {N,T}
    gs = size(g)
    for i in 1:N
        x = idx[i]
        if x <= r || x > gs[i] - r
            return false
        end
    end
    true
end

@inline function neigh(stencil :: Stencil{N}, g :: Grid{T,N}, idx :: CartesianIndex{N}) where {T,N}
    gx :: Array{T, N} = g.x

    if is_safe_neigh(idx.I, stencil.r, gx)
        return stencil.x + idx
    end

    clip(stencil.x + idx, size(gx), g.torus)
end

### Coarser Grid

struct CoarserGrid{T, N} <: AbstractGrid{T, N}
    x :: Grid{T, N}
    upscale :: Int
end

@inline Base.size(g :: CoarserGrid{T, N}) where {T,N} = size(g.x)
@inline Base.getindex(g :: CoarserGrid{T, N}, i) where {T,N} = getindex(g.x, i)
@inline Base.getindex(g :: CoarserGrid{T, N}, idxs...) where {T,N} = getindex(g.x, idxs...)

function nearby_positions(g :: CoarserGrid{T, N}, idx :: CartesianIndex{N}) where {T, N}
    s = size(g)
    u = g.upscale
    is = [idx.I...]

    lt = map(i -> div(i - 1, u) + 1, is)
    rb = lt .+ 1
    t = map(i -> mod(i - 1, u) / u, is)

    for i in 1:length(rb)
        if rb[i] > s[i]
            if g.x.torus[i]
                rb[i] = 1
            else
                rb[i] = s[i]
                t[i] = 0.5
            end
        end
    end

    lt, rb, t
end

function sort_pixs(lt, rb)
    if length(lt) == 1
        return [lt, rb]
    end

    ps_suffix = sort_pixs(lt[2:end], rb[2:end])

    ps = []
    for p in ps_suffix
        push!(ps, (lt[1], p...))
        push!(ps, (rb[1], p...))
    end

    ps
end

function nearby_val(g :: CoarserGrid{T, N}, idx :: CartesianIndex{N}) where {T, N}
    lt, rb, t = nearby_positions(g, idx)

    idxs = map(i -> CartesianIndex(i...), sort_pixs(lt, rb))
    vals = g.x.x[idxs]

    for i in 1:length(lt)
        new_vals = []
        for j in 1:2:length(vals)
            push!(new_vals, (1. - t[i]) * vals[j] + t[i] * vals[j+1])
        end
        vals = new_vals
    end

    vals[1]
end

#end
