# 1. allocate species from pools regarding states rules
# 2. apply rules from all states and produce new species to associated pools
# 3. redistribute species from pools by system transitions rules

import Base.eltype, Base.setproperty!, Base.getproperty, Base.identity
import LinearAlgebra.normalize

using LinearAlgebra, RecursiveArrayTools, SparseArrays, StaticArrays, Random,
    DifferentialEquations, Lazy, MultivariateStats, DEDataArrays

import Distributions, Distributions.Multinomial

using Pkg, Chain

get_pkg_version(name::AbstractString) =
    @chain Pkg.dependencies() begin
        values
        [x for x in _ if x.name == name]
        only
        _.version
    end

include("big_array_partition.jl")
include("rf_func.jl")
include("rf_math.jl")
include("rf_filters.jl")
include("rf_types.jl")
include("rf_preproc.jl")
include("rf_alloc.jl")
include("rf_rules.jl")
include("rf_trans.jl")
include("rf_init.jl")
include("rf_utils.jl")
include("rf_solver.jl")
include("rf_pi_dsl.jl")
include("rf_rule_lib.jl")
include("rf_agents.jl")
include("rf_optimize.jl")
