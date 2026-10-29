"""Idealized finger force sensing from all contacts, without object identity."""
import mujoco
import numpy as np

FINGERS=('thumb','index','middle','ring','pinky')
FEATURE_NAMES=[f'{finger}_{kind}_force_n' for finger in FINGERS for kind in ('normal','shear')]


def sense(model,data):
    values=np.zeros((5,2))
    for i in range(data.ncon):
        contact=data.contact[i];force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
        ids=[int(model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
        # Internal hand/robot collision loads are not fingertip tactile pads.
        if model.body_rootid[ids[0]]==model.body_rootid[ids[1]]:continue
        bodies={model.body(b).name for b in ids}
        for j,finger in enumerate(FINGERS):
            if any(name.startswith('R_'+finger+'_') and name.endswith(('distal','intermediate')) for name in bodies):
                values[j]+=np.array([max(0.,force[0]),np.linalg.norm(force[1:3])])
    return values.ravel()


class GripFeedback:
    """Bounded force trim and grasp-confirmation pause; no object state input."""
    def __init__(self,target_force=6.,enabled=True):
        self.target=target_force;self.enabled=enabled;self.trim=np.zeros(6);self.stable=0
        self.waited=0.;self.events=[];self.last_tactile=np.zeros(10);self.previous_base=None

    def __call__(self,t,base,model,data):
        tactile=sense(model,data);self.last_tactile=tactile.copy();forces=tactile.reshape(5,2)[:,0]
        ready=forces[0]>.1 and np.max(forces[1:])>.1
        self.stable=self.stable+1 if ready else 0
        result=base.copy();advance=True
        # Trim only the closed/carry part. Do not add closure during release.
        if self.enabled and 4.65<=t<10.1:
            mapping=((0,4),(1,3),(2,2)) # thumb pitch, index, middle
            for finger,motor in mapping:
                error=self.target-forces[finger]
                step=-min(.015,.001*(forces[finger]-15)) if forces[finger]>15 else np.clip(.002*error*.05,-.002,.002)
                # Once a pad touches, increasing reference closure must not
                # overwhelm pressure regulation. Opening is always permitted.
                if forces[finger]>.1 and self.previous_base is not None:
                    step-=max(0.,base[7+motor]-self.previous_base[7+motor])
                self.trim[motor]=np.clip(self.trim[motor]+step,-.12,.06)
            result[7:]+=self.trim
        else:self.trim*=.8
        if self.enabled and 4.8<=t<4.8+.05 and self.stable<3 and self.waited<1.:
            advance=False;self.waited+=.05
            self.events.append(dict(source_seconds=t,event='wait_for_thumb_and_opposing_contact'))
        self.previous_base=base.copy()
        return result,advance,tactile
