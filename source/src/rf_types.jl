abstract type AbstractSpecies end
abstract type AbstractPool{T,N} <: DEDataArray{T,N} end
abstract type AbstractRule end
abstract type AbstractSystem end
abstract type AbstractSolution end

struct Species <: AbstractSpecies
    idx :: Integer
    label
end

const PARTITION = Val{:type_partition}
const Condition = Function
const MIN_ALLOC_COND_REQ = 2.0 / typemax(Int)
const ALLOC_ALL_TYPE = Val{:alloc_all}()

const Continuous = Float64
const Discrete = Int64

const Tensor = AbstractArray
struct MemoryInt <: Integer end
struct MassActionContinuous <: Real end
const MassActionReal = MassActionContinuous

mutable struct Pool{T,N} <: AbstractPool{T,N}
    x :: Array{T,N}
    label
end

get_type(p :: Pool{T,N}) where {T,N} = T # change for eltype
eltype(p :: Pool{T,N}) where {T,N} = T
@inline get_data(p :: Pool{T,N}) where {T,N} = p.x
Base.convert(::Type{Array{T,N}}, p :: Pool{T,N}) where {T,N} = p.x
Base.size(p::Pool{T,N}) where {T,N} = size(p.x)
Base.getindex(p::Pool{T,N}, inds...) where {T,N} = getindex(p.x, inds...)

# array of SparseVector{Float64} where index i corresponds with i'th species of given type
const Stochiometry = ArrayPartition

struct GeneralRule{T, RS, ST} <: AbstractRule
    input :: ST
    output :: ST
    k :: RS # reaction speed
    label
end

const GeneralReactiveRule = GeneralRule{T, RS, Stochiometry} where {T, RS}

struct StructCoefs{T} end

get_reactants_stochiometry(r::GeneralReactiveRule{T, S}, ::Type{Val{:type_partition}}) where {T, S} = r.input.x
get_reactants_stochiometry(r::GeneralReactiveRule{T, S}, ::Type{Val{:type_partition}}) where {T <: StructCoefs{TT} where TT, S} = map(p -> map(first, p), r.input.x)

const FnArgs = Tuple{Function, Tuple}
const ReactiveRule{T} = GeneralReactiveRule{T, Number} where T
const FnReactiveRule = GeneralReactiveRule{T, Function} where T
const FnArgReactiveRule = GeneralReactiveRule{T, FnArgs} where T
const ContFnReactiveRule = GeneralReactiveRule{Continuous, Function}
const ContFnArgReactiveRule = GeneralReactiveRule{Continuous, FnArgs}

get_type(r :: ReactiveRule{T}) where T = T
eltype(r :: ReactiveRule{T}) where T = T

get_reactants_stochiometry(r::ReactiveRule{T}, ::Type{Val{:type_partition}}) where T = r.input.x
get_products_stochiometry(r::ReactiveRule{T}, ::Type{Val{:type_partition}}) where T = r.output.x
get_products_stochiometry(r::GeneralReactiveRule{T, RS}, ::Type{Val{:type_partition}}) where {T, RS} = r.output.x

get_k(r::ReactiveRule{T}) where T = r.k
get_k(r::GeneralReactiveRule{T, RS}) where {T, RS} = r.k

as_vector(a) = reshape(a, length(a))

struct NopRule <: AbstractRule
    label
end

struct ReactiveRuleGroup <: AbstractRule
    rules :: Vector{AbstractRule}
    label
end

mutable struct TimeDelayedRule <: AbstractRule
    rule :: AbstractRule
    T :: Real
    label
end

get_reactants_stochiometry(r::TimeDelayedRule, ::Type{Val{:type_partition}}) = get_reactants_stochiometry(r.rule, PARTITION)
get_products_stochiometry(r::TimeDelayedRule, ::Type{Val{:type_partition}}) = get_products_stochiometry(r.rule, PARTITION)

function get_reactants_stochiometry(r::ReactiveRuleGroup, ::Type{Val{:type_partition}})
    reduce((stoch, r) -> map((sa,s) -> maximum(hcat(sa, s), dims = 2) |> as_vector,
                             stoch, get_reactants_stochiometry(r, PARTITION)),
           r.rules,
           init = map(s -> fill(0, size(s)), get_reactants_stochiometry(r.rules[1], PARTITION)))
end

### Agents

abstract type AbstractAgent end

mutable struct Agent{TUID, T} <: AbstractAgent
    uid :: TUID
    state :: T
end

const Agents = Vector{AbstractAgent}

# MultiPool container of pools of different types
const MultiPool = ArrayPartition

struct StateDesc
    # indices of multi pools that state can allocate from and produce to
    multi_pools :: Vector{Int}
    # indices of rules that acts on this state
    rules :: Vector{Int}
    # vector of allocators type per rule (:prealloc_min, :prealloc_all, :alloc_all)
    rules_allocators
    # multi_pool x rule x species weights matrix for priority allocation
    rules_weights :: AbstractVector{AbstractVector{AbstractVector{Float64}}}
    # multi pool x species weight matrix for distribution of products
    species_sinks_weights
    # list of multi pools indices that are used for transition through graph connected to this state
    multi_pools_transition_in :: Vector{Int}
    multi_pools_transition_out :: Vector{Int}
end

mutable struct System <: AbstractSystem
    transitions :: Function

    multi_pools :: AbstractArray
    rules :: Vector{AbstractRule}
    species :: Vector{AbstractSpecies}

    states :: Vector{StateDesc}

    cache :: Dict{Symbol, Any}
end

mutable struct SystemOfSystems <: AbstractSystem
    multi_pools :: AbstractArray
    species :: Vector{AbstractSpecies}

    systems :: Vector{AbstractSystem}

    cache :: Dict{Symbol, Any}
end

function setproperty!(system :: SystemOfSystems, s :: Symbol, v)
    setfield!(system, s, v)

    if s == :multi_pools
        foreach(sy -> setproperty!(sy, s, v), system.systems)
    end
end

function setcache!(system :: System, k, v)
    system.cache[k] = v
end

function setcache!(system :: SystemOfSystems, k, v)
    system.cache[k] = v

    if k == :rules_state
        foreach(system.systems, v) do (sy, val)
            sy.cache[k] = val
        end
    end
end

mutable struct Solution{T, Tu, Tt, Ts, Tsystem, Tsolver, Tparams} <: AbstractSolution
    u :: Tu
    t :: Tt
    s :: Ts
    system :: Tsystem
    solver :: Tsolver
    params :: Tparams
end

Solution(:: T, u :: Tu, t :: Tt, s
         :: Ts, system :: Tsystem,
         solver :: Tsolver, params :: Tparams) where {T, Tu, Tt, Ts, Tsystem, Tsolver, Tparams} = Solution{T, Tu, Tt, Ts, Tsystem, Tsolver, Tparams}(u, t, s, system, solver, params)

neutral_el(x :: T) where {T <: Condition} = 0.
neutral_el(x :: Type{T}) where {T <: Condition} = 0.

neutral_el(x :: Type{T}) where {T <: Number} = convert(x, 0)
neutral_el(x :: T) where {T <: Number} = convert(typeof(x), 0)

neutral_el(x :: T) where {T <: Agents} = AbstractAgent[]
neutral_el(x :: Type{T}) where {T <: Agents} = AbstractAgent[]

neutral_el(x :: Array{T,N}) where {T, N} = fill(neutral_el(T), size(x))
neutral_el(x :: Type{Array{T,N}}) where {T, N} = Array{T}(undef, [0 for i in 1:N]...)

function need_copy(el)
    !(typeof(el) <: Number)
end

function deepfill(el, s)
    if need_copy(el)
        t = fill(el, s)
        foreach(i -> t[i] = copy(el), eachindex(t))
    else
        t = fill(el, s)
    end
    t
end

function empty_allocation_t(type :: Type{T}, reqs) where T
    map(rs -> deepfill(neutral_el(type), size(rs)), reqs)
end

function empty_allocation(mpool, reqs)
    map((p, rs) -> deepfill(neutral_el(p[1]), size(rs)), mpool.x, reqs)
end

eltype(::Type{Array{Agent{T},1}}) where T = T

struct Req{T,N,AT} <: DEDataArray{T,N} where {AT}
    x :: AbstractArray{T,N}
    alloc_type :: AT
end
