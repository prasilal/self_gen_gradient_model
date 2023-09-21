""" The activity multi background constraint implements the activity constraint of Potts models,
    but allows users to specify locations on the grid where lambda_act is different.
    See {@link activity.jl} for the normal version of this constraint.
"""

struct ActivityMultiBgParams
    """ strength of the activityconstraint per cellkind and per background.
    """
    lambda_act_mbg

    """ An array where each element represents a different background type.
        This is again an set of {@ArrayCoordinate}s of the pixels belonging to that backgroundtype. These pixels
        will have the LAMBDA_ACT_MBG value of that backgroundtype, instead of the standard value.
    """
    background_voxels
end

function set_bg_voxels(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                       voxels) where {T, Tdc, N, Trnd}
    voxels = voxels == nothing ?  model.cfg.params[:activity_mbg].background_voxels : nothing
    model.state.constraints_states[:activity_mbg] = voxels
end

function init_activity_mbg(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg}) where {T, Tdc, N, Trnd}
    init_activity(model)
    set_bg_voxels(model, nothing)
end

""" Method to compute the Hamiltonian for this constraint.
    @param {IndexCoordinate} sourcei - coordinate of the source pixel that tries to copy.
    @param {IndexCoordinate} targeti - coordinate of the target pixel the source is trying
    to copy into.
    @param {CellId} src_type - cellid of the source pixel.
    @param {CellId} tgt_type - cellid of the target pixel.
    @return {number} the change in Hamiltonian for this copy attempt and this constraint.
"""
function activity_mbg_delta_h(model :: GridModel{T, N, CPMState{Tdc, Trnd, N}, CPMCfg},
                              sourcei, targeti, src_type, tgt_type) where {T, Tdc, N, Trnd}
    a_params = model.cfg.params[:activity]
    ambg_params = model.cfg.params[:activity_mbg]
    state = model.state.constraints_states[:activity_mbg]

    max_act = 0
    lambda_act = 0

    src_kind = cell_kind(model, src_type)
    tgt_kind = cell_kind(model, tgt_type)
    bgindex1 = 1
    bgindex2 = 1

    for bgkind in 1:length(state)
        if contains(state[bgkind], sourcei)
            bgindex1 = bgkind
        end
        if contains(state[bgkind], targeti)
            bgindex2 = bgkind
        end
    end

    if (src_type != BGID)
        max_act = a_params.max_act[src_kind]
        lambda_act = ambg_params.lambda_act_mbg[src_kind][bgindex1]
    else
        max_act = a_params.max_act[tgt_kind]
        lambda_act = ambg_params.lambda_act_mbg[tgt_kind][bgindex2]
    end

    if max_act == 0 || lambda_act == 0
        return 0
    end

    lambda_act * (activity_at(a_param.act_mean, model, targeti)
                  - activity_at(a_param.act_mean, model, sourcei)) / max_act
end

const ActivityMultiBgCfg = mk_cfg(
    init_constraints_state = [init_activity_mbg],
    soft_constraints = [activity_mbg_delta_h],
    post_setpix_listeners = [activity_post_setpix_listener],
    post_mcs_listeners = [activity_post_mcs_listeners]
)
