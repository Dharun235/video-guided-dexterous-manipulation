"""Set coupled robot joints in a kinematic workspace; not a physics controller."""
import mujoco


def set_configuration(model,data,control):
    """Kinematic IK workspace only. Never called on the running rollout data."""
    for aid,value in enumerate(control):
        trn=model.actuator_trnid[aid,0]
        if model.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:
            data.qpos[model.jnt_qposadr[trn]]=value
        else:
            adr,num=model.tendon_adr[trn],model.tendon_num[trn]
            for wrap in range(adr,adr+num):
                jid=model.wrap_objid[wrap]
                data.qpos[model.jnt_qposadr[jid]]=value/num
    for eid in range(model.neq):
        if model.eq_type[eid]==mujoco.mjtEq.mjEQ_JOINT:
            dependent,driver=model.eq_obj1id[eid],model.eq_obj2id[eid]
            v=data.qpos[model.jnt_qposadr[driver]]
            data.qpos[model.jnt_qposadr[dependent]]=sum(model.eq_data[eid,k]*v**k for k in range(5))
    mujoco.mj_kinematics(model,data)
