# Functional tools

partial(f, args...) = (next_args...) -> f(args..., next_args...)
isneqv(x, y) = x != y
isneqv(x) = y -> x != y

struct Reduced{T}
    x :: T
end

function reduced(f :: F, col; init = nothing) where {F <: Function}
    ret = init

    next = iterate(col)
    if init == nothing && next != nothing
        (ret, state) = next
        next = iterate(col, state)
    end

    while next !== nothing
        (x, state) = next
        ret = f(ret, x)
        if typeof(ret) <: Reduced
            return ret.x
        end
        next = iterate(col, state)
    end

    return ret
end

# select_keys(Dict(:a => 1, :b => 2), [:a])
function select_keys(d :: Dict{K,V}, ks) where {K,V}
#    @>> d collect filter(kv -> kv[1] in ks) Dict{K,V}()
    reduce(keys(d), init = Dict{K,V}()) do nd, k
        k in ks ? (nd[k] = d[k]; nd) : nd
    end
end

identity(args...) = length(args) == 1 ? args(1) : args

Base.haskey(s :: Set{T}, k :: T) where {T} = k in s
Base.keys(s :: Set{T}) where {T} = s

function as_func(d :: Dict{K,V}) where {K,V}
    k -> get(d, k, nothing)
end
