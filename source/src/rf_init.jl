function init_rule_state(rule :: ReactiveRule{T}) where {T <: AbstractFloat}
    nothing
end

function init_rule_state(rule :: ReactiveRule{T}) where {T <: MassActionReal}
    nothing
end

function init_rule_state(rule :: ReactiveRule{T}) where {T <: Integer}
    []
end

function init_rule_state(rule :: ReactiveRule{T}) where {T <: MemoryInt}
    [[], nothing]
end

function init_rule_state(rule :: ReactiveRuleGroup)
    map(init_rule_state, rule.rules)
end

function init_rule_state(rule :: GeneralReactiveRule{S,T}) where {S, T}
    Dict{Any, Any}()
end

function init_rule_state(rule :: GeneralReactiveRule{S,T}) where {S, T <: Function}
    try
        rule.k(rule)
    catch _
        Dict{Any, Any}()
    end
end

function init_rule_state(rule :: GeneralReactiveRule{S,T}) where {S, T <: FnArgs}
    try
        rule.k[1](rule)
    catch _
        Dict{Any, Any}()
    end
end

function init_rule_state(rule :: TimeDelayedRule)
    Dict{Symbol, Any}(:state => init_rule_state(rule.rule))
end

function init_system!(system :: System)
    # reqs, ws, mpools_mask = build_requirements(system.rules, system.states, system.multi_pools)

    reqs = prepare_requirements(system.rules, system.states)
    ws = prepare_rules_weights(system.states, reqs, system.multi_pools)
    mpools_mask = prepare_rules_mask(system.states, system.multi_pools)
    rules_by_state = make_rules_by_state(system.states, system.rules)

    system.cache[:reqs] = reqs
    system.cache[:reqs_weights] = ws
    system.cache[:rules_mpools_mask] = mpools_mask
    system.cache[:production_masks] = make_production_masks(system.multi_pools, system.states)
    system.cache[:rules_by_state] = rules_by_state
    system.cache[:transition_mpools] = transition_mpools(system.states)
    system.cache[:rows_allocators] = make_rows_allocators(system.states)
    system.cache[:allocators_types] = Set(system.cache[:rows_allocators])

    rules_state = @>> rules_by_state map(first) map(init_rule_state)
    system.cache[:rules_state] = rules_state

    system.cache[:cond_cache] = Dict{String,Any}()

    system.cache[:multi_pools_tmp] = Dict()
end

function init_system!(system :: SystemOfSystems)
    system.cache[:rules_state] = []

    foreach(system.systems) do s
        init_system!(s)
        push!(system.cache[:rules_state], s.cache[:rules_state])
    end
end
