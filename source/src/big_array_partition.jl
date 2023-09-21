# Based on ArrayPartition from RecursiveArrayTools

# From Iterators.jl. Moved here since Iterators.jl is not precompile safe anymore.

# Concatenate the output of n iterators
struct BChain{T<:AbstractVector}
    xss::T
end

# iteratorsize method defined at bottom because of how @generated functions work in 0.6 now

Base.length(it::BChain) = sum(length, it.xss)

Base.eltype(::Type{BChain{T}}) where {T} = typejoin([eltype(t) for t in T.parameters]...)

function Base.iterate(it::BChain)
    i = 1
    xs_state = nothing
    while i <= length(it.xss)
        xs_state = iterate(it.xss[i])
        xs_state !== nothing && return xs_state[1], (i, xs_state[2])
        i += 1
    end
    return nothing
end

function Base.iterate(it::BChain, state)
    i, xs_state = state
    xs_state = iterate(it.xss[i], xs_state)
    while xs_state == nothing
        i += 1
        i > length(it.xss) && return nothing
        xs_state = iterate(it.xss[i])
    end
    return xs_state[1], (i, xs_state[2])
end

##########################
#### BigArrayPartition ###

struct BigArrayPartition{T} <:  AbstractVector{T} # DEDataArray{T,1}
    x :: Array{T,1}
end

@inline BigArrayPartition(f::F, N) where F<:Function = BigArrayPartition(map(f, 1:N))
BigArrayPartition(x...) = BigArrayPartition([x...,])

function BigArrayPartition(x::T, ::Type{Val{copy_x}}=Val{false}) where {T<:AbstractVector,copy_x}
    if copy_x
        return BigArrayPartition{T}(copy.(x))
    else
        return BigArrayPartition{T}(x)
    end
end

## similar array partitions

Base.similar(A::BigArrayPartition{T}) where {T} = BigArrayPartition{T}(similar.(A.x))

# ignore dims since array partitions are vectors
Base.similar(A::BigArrayPartition, dims::NTuple{N,Int}) where {N} = similar(A)

# similar array partition of common type
@inline function Base.similar(A::BigArrayPartition, ::Type{T}) where {T}
    N = npartitions(A)
    BigArrayPartition(i->similar(A.x[i], T), N)
end

# ignore dims since array partitions are vectors
Base.similar(A::BigArrayPartition, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = similar(A, T)

# similar array partition with different types
function Base.similar(A::BigArrayPartition, ::Type{T}, ::Type{S}, R::DataType...) where {T,S}
    N = npartitions(A)
    N != length(R) + 2 &&
        throw(DimensionMismatch("number of types must be equal to number of partitions"))

    types = (T, S, R...) # new types
    @inline function f(i)
        similar(A.x[i], types[i])
    end
    BigArrayPartition(f, N)
end

Base.copy(A::BigArrayPartition{T}) where {T} = BigArrayPartition{T}(copy.(A.x))

## zeros
Base.zero(A::BigArrayPartition{T}) where {T} = BigArrayPartition{T}(zero.(A.x))
# ignore dims since array partitions are vectors
Base.zero(A::BigArrayPartition, dims::NTuple{N,Int}) where {N} = zero(A)

## ones

# special to work with units
function Base.ones(A::BigArrayPartition)
    N = npartitions(A)
    out = similar(A)
    for i in 1:N
        fill!(out.x[i], oneunit(eltype(out.x[i])))
    end
    out
end

# ignore dims since array partitions are vectors
Base.ones(A::BigArrayPartition, dims::NTuple{N,Int}) where {N} = ones(A)

## vector space operations

for op in (:+, :-)
    @eval begin
        function Base.$op(A::BigArrayPartition, B::BigArrayPartition)
            Base.broadcast($op, A, B)
        end

        function Base.$op(A::BigArrayPartition, B::Number)
            Base.broadcast($op, A, B)
        end

        function Base.$op(A::Number, B::BigArrayPartition)
            Base.broadcast($op, A, B)
        end
    end
end

for op in (:*, :/)
    @eval function Base.$op(A::BigArrayPartition, B::Number)
        Base.broadcast($op, A, B)
    end
end

function Base.:*(A::Number, B::BigArrayPartition)
    Base.broadcast(*, A, B)
end

function Base.:\(A::Number, B::BigArrayPartition)
    Base.broadcast(/, B, A)
end

Base.:(==)(A::BigArrayPartition,B::BigArrayPartition) = A.x == B.x

## Functional Constructs

Base.mapreduce(f,op,A::BigArrayPartition) = mapreduce(f,op,(mapreduce(f,op,x) for x in A.x))
Base.any(f,A::BigArrayPartition) = any(f,(any(f,x) for x in A.x))
Base.any(f::Function,A::BigArrayPartition) = any(f,(any(f,x) for x in A.x))
function Base.copyto!(dest::AbstractArray,A::BigArrayPartition)
    @assert length(dest) == length(A)
    cur = 1
    @inbounds for i in 1:length(A.x)
        dest[cur:(cur+length(A.x[i])-1)] .= vec(A.x[i])
        cur += length(A.x[i])
    end
    dest
end

function Base.copyto!(A::BigArrayPartition,src::BigArrayPartition)
    @assert length(src) == length(A)
    if size.(A.x) == size.(src.x)
        A .= src
    else
        cnt = 0
        for i in eachindex(A.x)
            x = A.x[i]
            for k in eachindex(x)
                cnt += 1
                x[k] = src[cnt]
            end
        end
    end
    A
end

## indexing

@inline Base.firstindex(A::BigArrayPartition) = 1
@inline Base.lastindex(A::BigArrayPartition) = length(A)

Base.checkbounds(A :: BigArrayPartition, i) = 0 < i <= length(A) || throw(BoundsError(A.x, i))

@inline function Base.getindex(A::BigArrayPartition, i::Int)
    @boundscheck checkbounds(A, i)
    @inbounds for j in 1:length(A.x)
        i -= length(A.x[j])
        if i <= 0
            return A.x[j][length(A.x[j])+i]
        end
    end
end

@inline function Base.getindex(A::BigArrayPartition, i::Int, j...)
    @boundscheck 0 < i <= length(A.x) || throw(BoundsError(A.x, i))
    @inbounds b = A.x[i]
    @boundscheck checkbounds(b, j...)
    @inbounds return b[j...]
end

Base.getindex(A::BigArrayPartition{T}, ::Colon) where {T} = T[a for a in BChain(A.x)]

@inline function Base.setindex!(A::BigArrayPartition, v, i::Int)
    @boundscheck checkbounds(A, i)
    @inbounds for j in 1:length(A.x)
        i -= length(A.x[j])
        if i <= 0
            A.x[j][length(A.x[j])+i] = v
            break
        end
    end
end

@inline function Base.setindex!(A::BigArrayPartition, v, i::Int, j...)
    @boundscheck 0 < i <= length(A.x) || throw(BoundsError(A.x, i))
    @inbounds b = A.x[i]
    @boundscheck checkbounds(b, j...)
    @inbounds b[j...] = v
end

Base.iterate(A::BigArrayPartition) = iterate(BChain(A.x))
Base.iterate(A::BigArrayPartition,state) = iterate(BChain(A.x),state)

Base.length(A::BigArrayPartition) = sum((length(x) for x in A.x))
Base.size(A::BigArrayPartition) = (length(A),)

Base.first(A::BigArrayPartition) = first(first(A.x))
Base.last(A::BigArrayPartition) = last(last(A.x))

## broadcasting

struct BigArrayPartitionStyle{Style <: Broadcast.BroadcastStyle} <: Broadcast.AbstractArrayStyle{Any} end
BigArrayPartitionStyle(::S) where {S} = BigArrayPartitionStyle{S}()
BigArrayPartitionStyle(::S, ::Val{N}) where {S,N} = BigArrayPartitionStyle(S(Val(N)))
BigArrayPartitionStyle(::Val{N}) where N = BigArrayPartitionStyle{Broadcast.DefaultArrayStyle{N}}()

# promotion rules
@inline function Broadcast.BroadcastStyle(::BigArrayPartitionStyle{AStyle}, ::BigArrayPartitionStyle{BStyle}) where {AStyle, BStyle}
    BigArrayPartitionStyle(Broadcast.BroadcastStyle(AStyle(), BStyle()))
end
Broadcast.BroadcastStyle(::BigArrayPartitionStyle{Style}, ::Broadcast.DefaultArrayStyle{0}) where Style<:Broadcast.BroadcastStyle = BigArrayPartitionStyle{Style}()
Broadcast.BroadcastStyle(::BigArrayPartitionStyle, ::Broadcast.DefaultArrayStyle{N}) where N = Broadcast.DefaultArrayStyle{N}()

function Broadcast.combine_styles(A::BigArrayPartition{AbstractArray{T,N} where T}) where N
    return Broadcast.result_style(Broadcast.BroadcastStyle(eltype(A.x[1])), Broadcast.combine_styles(A.x[2:end]))
end

function Broadcast.combine_styles(A::BigArrayPartition{Any})
    return Broadcast.result_style(Broadcast.BroadcastStyle(eltype(A.x[1])), Broadcast.combine_styles(A.x[2:end]))
end

function Broadcast.BroadcastStyle(::Type{BigArrayPartition{T}}) where {T}
    Style = Broadcast.result_style(Broadcast.BroadcastStyle(T))
    BigArrayPartitionStyle(Style)
end

@inline function Base.copy(bc::Broadcast.Broadcasted{BigArrayPartitionStyle{Style}}) where Style
    N = npartitions(bc)
    @inline function f(i)
        copy(unpack(bc, i))
    end
    BigArrayPartition(f, N)
end

@inline function Base.copyto!(dest::BigArrayPartition, bc::Broadcast.Broadcasted{BigArrayPartitionStyle{Style}}) where Style
    N = npartitions(dest, bc)
    @inbounds for i in 1:N
        copyto!(dest.x[i], unpack(bc, i))
    end
    dest
end

## broadcasting utils

"""
    npartitions(A...)

Retrieve number of partitions of `ArrayPartitions` in `A...`, or throw an error if there are
`ArrayPartitions` with a different number of partitions.
"""

npartitions(A) = 0
npartitions(A::BigArrayPartition) = length(A.x)
npartitions(bc::Broadcast.Broadcasted) = _npartitions(bc.args)
npartitions(A, Bs...) = common_number(npartitions(A), _npartitions(Bs))

@inline _npartitions(args::Tuple) = common_number(npartitions(args[1]), _npartitions(Base.tail(args)))
_npartitions(args::Tuple{Any}) = npartitions(args[1])
_npartitions(args::Tuple{}) = 0

# drop axes because it is easier to recompute
@inline unpack(bc::Broadcast.Broadcasted{Style}, i) where Style = Broadcast.Broadcasted{Style}(bc.f, unpack_args(i, bc.args))
@inline unpack(bc::Broadcast.Broadcasted{BigArrayPartitionStyle{Style}}, i) where Style = Broadcast.Broadcasted{Style}(bc.f, unpack_args(i, bc.args))
unpack(x,::Any) = x
unpack(x::BigArrayPartition, i) = x.x[i]

@inline unpack_args(i, args::Tuple) = (unpack(args[1], i), unpack_args(i, Base.tail(args))...)
unpack_args(i, args::Tuple{Any}) = (unpack(args[1], i),)
unpack_args(::Any, args::Tuple{}) = ()

## utils
common_number(a, b) =
    a == 0 ? b :
    (b == 0 ? a :
     (a == b ? a :
      throw(DimensionMismatch("number of partitions must be equal"))))

function RecursiveArrayTools.recursive_unitless_bottom_eltype(A::BigArrayPartition{T}) where T
    reduce(A.x, init = Union{}) do t, x
        Union{t, RecursiveArrayTools.recursive_unitless_bottom_eltype(x)}
    end
end

function RecursiveArrayTools.recursive_unitless_eltype(A::BigArrayPartition{T}) where T
    reduce(A.x, init = Union{}) do t, x
        Union{t, RecursiveArrayTools.recursive_unitless_eltype(x)}
    end
end
