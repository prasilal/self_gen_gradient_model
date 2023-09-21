# Math utils

@inline function norm_cols(a)
    b = copy(a)
    for i in 1:size(a)[2]
        if dot(b[:,i], b[:,i]) != 0
            normalize!(@view(b[:,i]))
        end
    end
    b
end

@inline function norm_rows(a)
    b = copy(a)
    for i in 1:size(a)[1]
        if dot(b[i,:], b[i,:]) != 0
            normalize!(@view(b[i,:]))
        end
    end
    b
end

@inline function num_non_zero_cols(a)
    s = sum(a, dims = 1)
    @. s[s != 0] = 1
    sum(s)
end

@inline dist2(v1, v2) = (d = v1 .- v2; dot(d,d))
@inline norm2(v) = dot(v, v)
@inline wdist2(v1, v2, ws) = (d = v1 .- v2; d2 = d .* d .* ws'; sum(d2))
@inline normalize(t :: Tuple) = (n = norm(t); n != 0 ? t ./ n : t)

@inline sigmoid(x,l,k,x0) = l / (1 + exp(-k*(x-x0)))
@inline sigmoid_deriv(x,l,k,x0) = (y = sigmoid(x,l,k,x0); k * y * (1 - y))
@inline saturated(x,t) = sigmoid_deriv(x*t, 1., 1., 0.)

# Hill's dynamics
@inline hill_act(x, n, k, kA) = (xn = (x/kA)^n; k * xn / (1 + xn))
@inline hill_inh(x, n, k, kA) = (xn = (x/kA)^n; k / (1 + xn))

@inline hill_act(x, n, m, k, kA) = k * x^n / (kA^m + x^m)
@inline hill_inh(x, n, m, k, kA) = k * kA^n / (kA^m + x^m)

# Cartesian Indices
for op in (:+, :-)
    @eval begin
        @inline function Base.$op(idxs :: CartesianIndices{N, TT}, t :: NTuple{N, T}) where {N, T <: Int, TT}
            CartesianIndices(map((rng, offset) -> Base.broadcast($op, rng, offset), idxs.indices, t))
        end

        @inline function Base.$op(idxs :: CartesianIndices{N, TT}, t :: CartesianIndex{N}) where {N, TT}
            CartesianIndices(map((rng, offset) -> Base.broadcast($op, rng, offset), idxs.indices, t.I))
        end

        @inline function Base.$op(idx :: CartesianIndex{N}, t :: NTuple{N, T}) where {N, T <: Int, TT}
            CartesianIndex(Base.broadcast($op, idx.I, t))
        end

        @inline function Base.$op(idx :: CartesianIndex{N}, t :: CartesianIndex{N}) where {N}
            CartesianIndex(Base.broadcast($op, idx.I, t.I))
        end

        @inline function Base.$op(idxs :: Array{CartesianIndex{N}, M}, t :: NTuple{N, T}) where {N, M, T <: Int, TT}
            # map(i -> ($op)(i, t), idxs)
            l = length(idxs)
            new_idxs = Array{CartesianIndex{N}}(undef, l)
            @inbounds for i in 1:l
                new_idxs[i] = ($op)(idxs[i], t)
            end
            new_idxs
        end

        @inline function Base.$op(t :: NTuple{N, T}, idxs :: Array{CartesianIndex{N}, M}) where {N, M, T <: Int, TT}
            # map(i -> ($op)(t, i), idxs)
            l = length(idxs)
            new_idxs = Array{CartesianIndex{N}}(undef, l)
            @inbounds for i in 1:l
                new_idxs[i] = ($op)(t, idxs[i])
            end
            new_idxs
        end

        @inline function Base.$op(idxs :: Array{CartesianIndex{N}, M}, t :: CartesianIndex{N}) where {N, M}
            # map(i -> ($op)(i, t), idxs)
            l = length(idxs)
            new_idxs = Array{CartesianIndex{N}}(undef, l)
            @inbounds for i in 1:l
                new_idxs[i] = ($op)(idxs[i], t)
            end
            new_idxs
        end

        @inline function Base.$op(t :: CartesianIndex{N}, idxs :: Array{CartesianIndex{N}, M}) where {N, M}
            # map(i -> ($op)(t, i), idxs)
            l = length(idxs)
            new_idxs = Array{CartesianIndex{N}}(undef, l)
            @inbounds for i in 1:l
                new_idxs[i] = ($op)(t, idxs[i])
            end
            new_idxs
        end

        @inline function Base.$op(idx :: CartesianIndex{N}, t :: Array{T, 1}) where {T <: Number, N}
            CartesianIndex(Base.broadcast($op, idx.I, t))
        end

        @inline function Base.$op(t :: NTuple{N, T}, i :: Array{TT, 1}) where {N, T, TT <: Number}
            Base.broadcast($op, t, i)
        end

        @inline function Base.$op(i :: Array{TT, N}, t :: NTuple{N, T}) where {N, T, TT <: Number}
            Base.broadcast($op, i, t)
        end

        @inline function Base.$op(t :: Array{T, 1}, idx :: CartesianIndex{N}) where {T <: Number, N}
            CartesianIndex(Base.broadcast($op, t, idx.I))
        end
    end
end

@inline or(x :: Bool, y :: Bool) = x || y
@inline and(x :: Bool, y :: Bool) = x && y

@inline function valid_idx(idx :: CartesianIndex{N}, rng :: NTuple{N, T}, round :: NTuple{N, Bool}) where {N, T <: Int}
    @inbounds for (i, pidx) in enumerate(idx.I)
        if pidx < 1 || pidx > rng[i]
            if round[i]
                return 0
            else
                return -1
            end
        end
    end

    return 1
end

@inline function clamp(i :: T1, rng :: T2) where {T1 <: Int, T2 <: Int}
    if i < 1
        return rng + i
    elseif i > rng
        return i - rng
    end

    return i
end

@inline function clip(idxs :: Array{CartesianIndex{N}, M}, rng :: NTuple{N, T}, round :: NTuple{N, Bool}) where {M, N, T <: Int}
    # new_idxs = CartesianIndex{N}[]
    new_idxs = Array{CartesianIndex{N}}(undef, 0)
    new_idx = Array{Int}(undef, N)

    @inbounds for i in 1:length(idxs)
        idx = idxs[i]
        res = valid_idx(idx, rng, round)

        if res == 1
            push!(new_idxs, idx)
        elseif res == 0
            for j in 1:N
                new_idx[j] = clamp(idx.I[j], rng[j])
            end

            # t = tuple(new_idx...)
            t = SVector{N, Int}(new_idx).data
            ni = CartesianIndex(t)
            push!(new_idxs, ni)
            # push!(new_idxs, CartesianIndex(map((i, l) -> clamp(i, l), idx.I, rng)))
        end
    end

    new_idxs
end
