## RF Agents utilities
using UUIDs

gen_agent_unique_id() = uuid4()

abstract type AbstractAgentShape end
abstract type AbstractAgentState end

mutable struct SphereShape <: AbstractAgentShape
    pos :: Tuple{Vararg{Real}}
    radius :: Real
    attrs :: Dict{Symbol, Any}
end

get_pos(s :: SphereShape) = s.pos
get_pos_int(s :: SphereShape) = convert.(Int, floor.(s.pos))
get_attrs(s :: SphereShape) = s.attrs
set_pos!(s :: SphereShape, pos :: Tuple) = s.pos = pos
set_pos!(s :: SphereShape, pos :: Vector{T}) where {T <: Real} = s.pos = tuple(pos...)
get_shape(s :: SphereShape) = ball_indices(length(s.pos), s.radius)
get_neighborhood(s :: SphereShape) = ball_surface_indices(length(s.pos), s.radius + 1, s.radius)
get_bounding_sphere(s :: SphereShape) = s.radius

function shapes_intersect_pot(s1 :: SphereShape, s2 :: SphereShape,
                              pos1 :: Vector{T}, pos2 :: Vector{T}) where {T <: Real}
    sigmoid((s1.radius + s2.radius)^2 - dist2(pos1, pos2), 1.0, 100.0, 0.0)
end

mutable struct AgentWithSystemAndShape{SH,SY} <: AbstractAgentState where {SH <: AbstractAgentShape, SY <: AbstractAgentState}
    shape :: SH
    system :: SY
end

function mk_agent_with_system_n_shape(id, shape, def_system :: Function)
    Agent(id, AgentWithSystemAndShape(shape, def_system()))
end

get_agent_system(a :: Agent{TUID, AgentWithSystemAndShape{SH, SY}}) where {TUID, SH, SY} = a.state.system
get_agent_shape(a :: Agent{TUID, AgentWithSystemAndShape{SH, SY}}) where {TUID, SH, SY} = a.state.shape

abstract type AbstractEvent end

struct Event{T,A} <: AbstractEvent
    type :: T
    args :: A
end

gen_ev(tp, args) = Agent(gen_agent_unique_id(), Event(tp,args))
get_ev(a :: Agent{TUID, Event{T,A}}) where {TUID, T,A} = a.state

##################################
## Agent with inner system Rules #
##################################

"""
Agent state:
    * :: System
    * :: Shape
        * get_pos -> CartesianIndex of center
        * get_shape -> Vector{CartesianIndex} vector of relative indices to center defining the shape
        * get_neighborhood -> Array{CartesianIndex} array of relative indices to center defining the shape
"""

@inline shape_positioned(idx, pos) = map(i -> CartesianIndex(i.I .+ pos), idx)

@inline function shape_centered(shape :: SphereShape)
    shape_positioned(get_shape(shape), get_pos_int(shape))
end

@inline function neighborhood_centered(shape :: SphereShape)
    shape_positioned(get_neighborhood(shape), get_pos_int(shape))
end

function comp_agent_masks(out)
    masks = [fill(false, length(out[i])) for i in 1:length(out)]
    foreach_item(out) do i, j
        if out[i][j] != nothing && out[i][j] != 0
            mask_idx = out[i][j]
            masks[mask_idx[1]][mask_idx[2]] = true
        end
    end
    masks
end

function rule_agent_build_masks(rule, _, reqs, allocated, _, produced, used, state, t, system)
    out = rule.output.x
    masks = comp_agent_masks(out)

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0 && masks[i][j]
            produced[i][j] .= reqs[i][j]
            used[i][j] .= allocated[i][j]
        end
    end

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0 && !masks[i][j]
            mask_idx = out[i][j]
            mask = produced[mask_idx[1]][mask_idx[2]]
            foreach(allocated[i][j]) do agent
                # shape_abs = shape_centered(agent.state.shape) # ? TODO: maybe do it inplace
                shape_abs = shape_centered(get_agent_shape(agent)) # ? TODO: maybe do it inplace
                mask[shape_abs] .= reqs[i][j]
            end
        end
    end
end

@inline function gather_resources(reqs, out)
    to_be_allocated = []

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0 && out[i][j] == 0
            push!(to_be_allocated, (i,j))
        end
    end

    to_be_allocated
end

@inline is_all_alloc_q(state_desc) = state_desc[:alloc_type] == ALLOC_ALL_TYPE

@inline function append_allocated(is_all_alloc, allocated, produced, used, i, j)
    if !is_all_alloc
        append!(produced[i][j], allocated[i][j])
        append!(used[i][j], allocated[i][j])
    end
end

# reqs != 0 and out == 0 allocated resource that will be moved to inner agent system
# reqs != 0 and out != 0 inner agent systems, agents out represent mpool id
function rule_agent_alloc_outer(rule, state_desc, reqs, allocated, _, produced, used, state, t, system)
    out = rule.output.x
    is_all_alloc = is_all_alloc_q(state_desc)

    to_be_allocated = gather_resources(reqs, out)

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0 && out[i][j] != 0
            append_allocated(is_all_alloc, allocated, produced, used, i, j)
            foreach(allocated[i][j]) do agent
                # nb_abs = neighborhood_centered(agent.state.shape)
                nb_abs = neighborhood_centered(get_agent_shape(agent))
                # mpools = agent.state.system.multi_pools.x
                mpools = get_agent_system(agent).multi_pools.x
                foreach(to_be_allocated) do (k,l)
                    mpools[out[i][j]].x[k].x[l] = allocated[k][l][nb_abs]
                end
            end
        end
    end
end

# reqs != 0 and out == 0 resources that will be converted from req mpool to out mpool
# reqs != 0 and out != 0 agent systems
function rule_agent_alloc_mpool_convert(convert_fn, rule, state_desc, reqs, allocated, _, produced, used, state, t, system)
    out = rule.output.x
    is_all_alloc = is_all_alloc_q(state_desc)

    to_be_converted = gather_resources(reqs, out)

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0 && out[i][j] != 0
            append_allocated(is_all_alloc, allocated, produced, used, i, j)
            foreach(allocated[i][j]) do agent
                mpools = agent.state.system.multi_pools.x
                foreach(to_be_converted) do (k,l)
                    mpools[out[i][j]].x[k].x[l] = convert_fn(mpools[reqs[i][j]].x[k].x[l])
                end
            end
        end
    end
end

@def_fn_rule rule_agent_system_iterate begin
    is_all_alloc = is_all_alloc_q(state_desc)

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0
            append_allocated(is_all_alloc, allocated, produced, used, i, j)
            foreach(allocated[i][j]) do agent
                sol = init_solver(Val{:simple}, system = agent.state.system, tspan = (0, t), ts = t)
                solve!(sol, save_states = false)
            end
        end
    end
end

@def_fn_rule rule_agent_produce begin
    out = rule.output.x

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0
            foreach(allocated[i][j]) do agent
                # nb_abs = neighborhood_centered(agent.state.shape)
                nb_abs = neighborhood_centered(get_agent_shape(agent))
                # mpools = agent.state.system.multi_pools.x
                mpools = get_agent_system(agent).multi_pools.x
                foreach_item(out) do k,l
                    if out[k][l] != 0
                        produced[k][l][nb_abs] .+= mpools[out[k][l]].x[k].x[l] ./ length(nb_abs)
                    end
                end
            end
        end
    end
end

@def_fn_rule rule_agent_foreach_match_action_resolve begin
    match, match_args, act, act_args, resolve, resolve_args = rule_args(rule)
    is_all_alloc = is_all_alloc_q(state_desc)
    out = rule.output.x

    foreach_item(reqs) do i,j
        if reqs[i][j] != 0
            append_allocated(is_all_alloc, allocated, produced, used, i, j)
            act_res = []
            foreach(allocated[i][j]) do agent
                res = @add_call_rule_args match(agent, out[i][j], match_args...)
                push!(act_res, @add_call_rule_args act(agent, res, act_args...))
            end
            @add_call_rule_args resolve(allocated[i][j], act_res, resolve_args...)
        end
    end
end

@def_fn_rule match_evt begin
    mp, p, s = params
    system = get_agent_system(agent)
    evs = get_sys_mpool(system, mp)[p].x[s]

    i = findfirst(a -> get_ev(a).type == ev_tp, evs)

    get_ev(evs[i]).args
end agent params ev_tp

# TODO: we have to do it globally we need to solve intersections so use map -> reduce
@def_fn_rule agent_move begin
    system = get_agent_system(agent)
    sh = agent |> get_agent_shape
    pos = get_pos(sh)

    if match == nothing
        return pos
    end

    (p, si), ds = match
    s = get_sys_mpool(system, mp)[p].x[si]
    _, i = findmax(s)
    center = shape_centered(sh)[i].I
    new_pos = @. ds * center + (1 - ds) * pos

    new_pos
end agent match mp

@def_fn_rule agent_intersection_resolve begin
    shapes = map(get_agent_shape, agents)
    new_pos = vec_of_vec_to_mtx(new_pos)

    res_pos = minimize_shapes_intersections(shapes, new_pos,
                                            position_unit_pot = position_unit_pot,
                                            angle_unit_pot = angle_unit_pot,
                                            collision_unit_pot = collision_unit_pot)

    for (i, agent) in agents |> enumerate
        set_pos!(get_agent_shape(agent), res_pos[:,i])
    end
end agents new_pos position_unit_pot angle_unit_pot collision_unit_pot

@def_fn_rule agent_expose_spec begin
    system = get_agent_system(agent)

    map(specs) do (mp, p, si)
        get_sys_mpool(system, mp)[p].x[si]
    end
end agent match specs

@inline get_agent_radius(a) = a |> get_agent_shape |> get_bounding_sphere
@inline get_agent_pos(a) = a |> get_agent_shape |> get_pos

function agents_nearby(a1, a2, max_gap)
    p1 = get_agent_pos(a1)
    p2 = get_agent_pos(a1)
    r1 = get_agent_radius(a1)
    r2 = get_agent_radius(a2)

    (r1 + r2 + max_gap)^2 >= dist2(p1, p2)
end

function update_pool_item(p :: Pool{Array{T,NN},N}, idx :: Int, v :: Array{T,NN}) where {T <: Real, NN ,N}
    if isempty(p.x[idx])
        p.x[idx] = v
    else
        p.x[idx] .+= v
    end
end

function update_agent_spec(a, ospecs, ispecs, ratio, dir)
    system = get_agent_system(a)
    shape = get_agent_shape(a)
    nb = get_neighborhood(shape)

    _, i = @>> nb map(i -> dot(normalize(i.I), dir)) findmax

    foreach(ospecs, ispecs) do os, is
        mp, p, si = is
        sps = zeros(length(nb))
        sps[i] = os * ratio
        update_pool_item(get_sys_mpool(system, mp)[p], si, sps)
    end
end

function agents_contact_ratio(a1, a2)
    r1 = get_agent_radius(a1)
    r2 = get_agent_radius(a2)

    0.25 * r2 / (r1 + r2)
end

function aget_contact_dir(a1, a2)
    dir = get_agent_pos(a2) .- get_agent_pos(a1)
    normalize(dir)
end

@def_fn_rule agent_gather_exposed_spec begin
    shapes = map(get_agent_shape, agents)
    pos = @>> shapes map(get_pos) vec_of_vec_to_mtx
    tree = BallTree(pos)
    r = 2. * (map(get_bounding_sphere, shapes) |> maximum) + max_gap

    for i in 1:length(agents)
        nbs = inrange(tree, pos[:,i], r, false)
        for j in nbs
            if i != j && agents_nearby(agents[i], agents[j], max_gap)
                ratio = contact_c * agents_contact_ratio(agents[i], agents[j])
                dir = aget_contact_dir(agents[i], agents[j])
                update_agent_spec(agents[i], outer_specs[j], inner_specs, ratio, dir)
            end
        end
    end

end agents outer_specs inner_specs max_gap contact_c
