""" Implements a data structure with constant-time insertion, deletion, and random
sampling. That's crucial for the CPM metropolis algorithm, which repeatedly needs to sample
pixels at cell borders. Elements in this set must be unique.
"""
#module DiceSets
#export AbstractDiceSet, DiceSet, push!, remove!, contains, sample

abstract type AbstractDiceSet end

mutable struct DiceSet{T, TRnd} <: AbstractDiceSet
    """ Map used to check in constant time whether a pixel is at the
    cell border. Keys are the actual values stored in the DiceSet, numbers are their
    location in the elements arrray.
    """
    indices :: Dict{T, Integer}
    """ Use an array for constant time random sampling of pixels at the border of cells. """
    elements :: Vector{T}
    """ The number of elements currently present in the DiceSet. """
    # len :: Integer
    rng :: TRnd

    """ The constructor of DiceSet takes a RNG object as input, to allow
    seeding of the random number generator used for random sampling.
    @param {TRnd} rng, RNG object used for random numbers.
    """
    # DiceSet{T, TRnd}(rng :: TRnd) where {T, TRnd} = new(Dict{T,Integer}(), Vector{T}(), 0, rng)
    DiceSet{T, TRnd}(rng :: TRnd) where {T, TRnd} = new(Dict{T,Integer}(), Vector{T}(), rng)
end

""" Insert a new element. It is added as an index in the indices, and pushed
to the end of the elements array.
@param {DiceSet{T, TRnd}} dc, DiceSet to insert
@param {T} v, The element to add.
"""
function Base.push!(ds :: DiceSet{T, TRnd}, v :: T) where {T, TRnd}
    indices :: Dict{T, Integer} = ds.indices
    if (haskey(indices, v))
        return ds;
    end

    """ Add element to both the hash map and the array. """
    # ds.indices[v] = ds.len + 1
    push!(ds.elements, v)
    indices[v] = length(ds.elements)
    # ds.len += 1

    return ds
end

""" Remove element v.
@param {DiceSet{T, TRnd}} dc, DiceSet to insert
@param {T} v, The element to remove.
"""
function Base.delete!(ds :: DiceSet{T, TRnd}, v :: T) where {T, TRnd}
    indices :: Dict{T, Integer} = ds.indices

    if (!haskey(indices, v))
        return ds
    end

    elements :: Vector{T} = ds.elements

    """ The dict gives the index in the array of the value to be removed.
    The value is removed directly from the dict, but from the array we
    initially remove the last element, which we then substitute for the
    element that should be removed. """
    i = indices[v]
    delete!(indices, v)
    e = pop!(elements)
    # ds.len -= 1
    if (e == v)
        return ds
    end

    elements[i] = e
    indices[e] = i

    return ds
end

""" Check if the DiceSet already contains element v.
@param {DiceSet{T, TRnd}} dc, DiceSet to insert
@param {T} v, The element to check presence of.
@return {boolean} true or false depending on whether the element is present or not.
"""
contains(ds :: DiceSet{T, TRnd}, v :: T) where {T, TRnd} = haskey(ds.indices, v)

""" Sample a random element from DiceSet.
@return the element sampled. """
sample(ds :: DiceSet{T, TRnd}) where {T, TRnd} = rand(ds.rng, ds.elements)

#=
import Random

ds = DiceSet{Int, Random.MersenneTwister}(Random.MersenneTwister(1234))

push!(ds, 1)
push!(ds, 2)
push!(ds, 3)
push!(ds, 4)

delete!(ds, 1)
contains(ds, 1)
contains(ds, 4)
sample(ds)
=#

#end
