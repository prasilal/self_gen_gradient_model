# Filters
using EasyRanges
using OffsetArrays

crop(i :: Int, m :: Int) = (i + m - 1) % m + 1
crop(idx, max_idx) = CartesianIndex(crop.(idx.I, max_idx.I))

if (get_pkg_version("LocalFilters").major < 2)
begin
using LocalFilters

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      M::AbstractArray{TM,N},
                      B::LocalFilters.Kernel{K,N},
                      initial::Function,
                      update::Function,
                      store::Function) where {T,TM,K,N}
    R = LocalFilters.cartesian_region(A)
    imin, imax = LocalFilters.limits(R)
    kmin, kmax = LocalFilters.limits(B)
    ker, off = LocalFilters.coefs(B), LocalFilters.offset(B)
    @inbounds for i in R
        v = initial(A[i], M[i], ker[off])
        k = i + off
        @simd for j in LocalFilters.cartesian_region(max(imin, i - kmax),
                                                     min(imax, i - kmin))
            v = update(v, A[j], M[j], ker[k - j])
        end
        store(dst, i, v)
    end
    return dst
end

function localfilter_torus!(dst::AbstractArray{T,N},
                            A::AbstractArray{T,N},
                            M::AbstractArray{TM,N},
                            B::LocalFilters.Kernel{K,N},
                            initial::Function,
                            update::Function,
                            store::Function) where {T,TM,K,N}
    R = LocalFilters.cartesian_region(A)
    imin, imax = LocalFilters.limits(R)
    kmin, kmax = LocalFilters.limits(B)
    ker, off = LocalFilters.coefs(B), LocalFilters.offset(B)
    @inbounds for i in R
        v = initial(A[i], M[i], ker[off])
        k = i + off
        l = i - kmax
        h = i - kmin
        not_border = max(imin, l) == l && min(imax, h) == h
        if not_border
            @simd for j in LocalFilters.cartesian_region(l, h)
                v = update(v, A[j], M[j], ker[k - j])
            end
        else
            @simd for j in LocalFilters.cartesian_region(l, h)
                jj = crop(j, imax)
                v = update(v, A[jj], M[jj], ker[k - j])
            end
        end
        store(dst, i, v)
    end
    return dst
end

@inline function mask_setter!(arr::AbstractArray, idx, val)
    (a,m,k) = val[3]
    arr[idx] = m*(val[1] - a*(val[2] - (m+1)*k))
end

@inline smooth_init!(a,m,k) = (0., 0., (a,m,k))
@inline smooth_update!(v,a,m,b) = (v[1] + a*m*b, v[2] + m*b, v[3])

function mask_smooth!(dst::AbstractArray{Td,N},
                      A::AbstractArray{Ts,N},
                      M::AbstractArray{Tm,N},
                      B::LocalFilters.Kernel{Tk,N}) where {Td,Ts,Tm,Tk,N}
    @assert axes(dst) == axes(A) == axes(M)
    #T = LocalFilters._typeofsum(promote_type(Ts, Tk, Tm))
    localfilter!(dst, A, M, B,
                 smooth_init!, # (a,m,k)   -> (zero(T), zero(T), (a,m,k)),
                 smooth_update!, # (v,a,m,b) -> (v[1] + a*m*b, v[2] + m*b, v[3])
                 mask_setter!)
end

function mask_smooth_torus!(dst::AbstractArray{Td,N},
                            A::AbstractArray{Ts,N},
                            M::AbstractArray{Tm,N},
                            B::LocalFilters.Kernel{Tk,N}) where {Td,Ts,Tm,Tk,N}
    @assert axes(dst) == axes(A) == axes(M)
    # T = LocalFilters._typeofsum(promote_type(Ts, Tk, Tm))
    localfilter_torus!(dst, A, M, B,
                       smooth_init!, # (a,m,k)   -> (zero(T), zero(T), (a,m,k)),
                       smooth_update!, # (v,a,m,b) -> (v[1] + a*m*b, v[2] + m*b, v[3]),
                       mask_setter!)
end

end # begin
end # if

#=

A = zeros(Float64, 10, 10)
M = ones(Float64, 10, 10)
B = OffsetArray([0 1 0; 1 0 1; 0 1 0], -1:1,-1:1)
C = zeros(Float64, 10, 10)

A[5,5] = 1.0

masked_manifold_smoothen!(C, A, M, B)

=#

function masked_manifold_smoothen!(dst :: AbstractArray{T,N},
                                   A :: AbstractArray{T,N},
                                   M :: AbstractArray{TM,N},
                                   B :: AbstractArray{K,N}) where {T,TM,K,N}
    TT = promote_type(eltype(A), eltype(B))

    imin = first(CartesianIndices(A))
    imax = last(CartesianIndices(A))

    lastb = last(CartesianIndices(B))
    firstb = first(CartesianIndices(B))

    @inbounds for i in CartesianIndices(dst)
        w = s = zero(TT)
        j_first = i + firstb
        j_last = i + lastb

        not_border = max(imin, j_first) == j_first && min(imax, j_last) == j_last

        if not_border
            @simd for j in j_first:j_last
                ww = M[j]*B[j-i]
                s += ww*A[j]
                w += ww
            end
        else
            @simd for j in j_first:j_last
                jj = crop(j, imax)
                ww = M[jj]*B[j-i]
                s += ww*A[jj]
                w += ww
            end
        end

        dst[i] = M[i]*(s - w*A[i])
    end
end


const laplacian_1d = [1 -2 1]
const laplacian_2d_5_stencil = [0 1 0; 1 -4 1; 0 1 0]
const laplacian_2d_9_stencil = [0.25 0.5 0.25; 0.5 -3 0.5; 0.25 0.5 0.25]
const laplacian_3d_7_stencil = cat([0 0 0; 0 1 0; 0 0 0],
                                   [0 1 0; 1 -6 1; 0 1 0],
                                   [0 0 0; 0 1 0; 0 0 0],
                                   dims=3)
const laplacian_3d_27_stencil = cat(1/26 * [2 3 2; 3 6 3; 2 3 2],
                                    1/26 * [3 6 3; 6 -88 6; 3 6 3],
                                    1/26 * [2 3 2; 3 6 3; 2 3 2],
                                    dims=3)

const masked_laplacian_1d = [1 0 1]
const masked_laplacian_2d_5_stencil = [0 1 0; 1 0 1; 0 1 0]
const masked_laplacian_2d_9_stencil = [0.25 0.5 0.25; 0.5 0 0.5; 0.25 0.5 0.25]
const masked_laplacian_3d_7_stencil = cat([0 0 0; 0 1 0; 0 0 0],
                                          [0 1 0; 1 0 1; 0 1 0],
                                          [0 0 0; 0 1 0; 0 0 0],
                                          dims=3)
const masked_laplacian_3d_27_stencil = cat(1/26 * [2 3 2; 3 6 3; 2 3 2],
                                           1/26 * [3 6 3; 6 0 6; 3 6 3],
                                           1/26 * [2 3 2; 3 6 3; 2 3 2],
                                           dims=3)

function masked_laplacian(dims; complex = false)
    if dims == 1
        masked_laplacian_1d
    elseif dims == 2
        (complex ? masked_laplacian_2d_9_stencil : masked_laplacian_2d_5_stencil)
    elseif dims == 3
        (complex ? masked_laplacian_3d_27_stencil : masked_laplacian_3d_7_stencil)
    else
        nothing
    end
end

function strictfloor(::Type{T}, x)::T where {T}
    n = floor(T, x)
    return (n < x ? n : n - one(T))
end

function ball_indices(rank::Integer, radius::Real)
    b = radius + 1/2
    r = strictfloor(Int, b)
    dim = 2*r + 1
    dims = ntuple(d->dim, rank)
    arr = Array{CartesianIndex}(undef, 0)
    qmax = strictfloor(Int, b^2)
    _ball!(arr, 0, qmax, r, 1:dim, Base.tail(dims))
    return arr
end

@inline function _ball!(arr::AbstractArray{CartesianIndex,N},
                        q::Int, qmax::Int, r::Int,
                        range::AbstractUnitRange{Int},
                        dims::Tuple{Int, Vararg{Int}}, I::Int...) where {N}
    nextdims = Base.tail(dims)
    x = -r
    for i in range
        # _ball!(arr, q + x*x, qmax, r, range, nextdims, I..., i)
        _ball!(arr, q + x*x, qmax, r, range, nextdims, I..., x)
        x += 1
    end
end

@inline function _ball!(arr::AbstractArray{CartesianIndex,N},
                        q::Int, qmax::Int, r::Int,
                        range::AbstractUnitRange{Int},
                        ::Tuple{}, I::Int...) where {N}
    x = -r
    for i in range
        if (q + x*x ≤ qmax)
            push!(arr,CartesianIndex(I...,x))
        end
        x += 1
    end
end

function ball_surface_indices(rank::Integer, max_radius::Real, min_radius::Real)
    b = max_radius + 1/2
    minb = min_radius + 1/2
    r = strictfloor(Int, b)
    dim = 2*r + 1
    dims = ntuple(d->dim, rank)
    arr = Array{CartesianIndex}(undef, 0)
    qmax = strictfloor(Int, b^2)
    qmin = strictfloor(Int, minb^2)
    _ball_surface!(arr, 0, qmax, qmin, r, 1:dim, Base.tail(dims))
    return arr
end

@inline function _ball_surface!(arr::AbstractArray{CartesianIndex,N},
                                q::Int, qmax::Int, qmin::Int, r::Int,
                                range::AbstractUnitRange{Int},
                                dims::Tuple{Int, Vararg{Int}}, I::Int...) where {N}
    nextdims = Base.tail(dims)
    x = -r
    for i in range
        _ball_surface!(arr, q + x*x, qmax, qmin, r, range, nextdims, I..., x)
        x += 1
    end
end

@inline function _ball_surface!(arr::AbstractArray{CartesianIndex,N},
                                q::Int, qmax::Int, qmin::Int, r::Int,
                                range::AbstractUnitRange{Int},
                                ::Tuple{}, I::Int...) where {N}
    x = -r
    for i in range
        qq = q + x*x
        if (qq ≤ qmax && qq >= qmin)
            push!(arr,CartesianIndex(I...,x))
        end
        x += 1
    end
end
