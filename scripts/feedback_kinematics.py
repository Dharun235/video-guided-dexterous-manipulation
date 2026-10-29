"""Add robot-only wrist Jacobians for physically scaled residual training."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main(args):
    schema=json.loads((args.reference/'dataset.json').read_text())
    model=mujoco.MjModel.from_xml_path(str((args.reference/'scene.xml').resolve()));data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,np.load(args.reference/'initial_state.npz')['integration'],mujoco.mjtState.mjSTATE_INTEGRATION)
    qadr=np.array(schema['state_qpos_indices']);aids=np.array(schema['active_actuator_indices'][:7])
    dofs=model.jnt_dofadr[model.actuator_trnid[aids,0]];site=model.site('landmark_0').id
    with np.load(args.cache/'cache.npz') as source:values={k:source[k] for k in source.files}
    for split in ('train','validation'):
        matrices=[]
        for row in values[split+'_features']:
            data.qpos[qadr]=row[:19];mujoco.mj_forward(model,data)
            jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(model,data,jp,jr,site)
            matrices.append(jp[:,dofs])
        values[split+'_jacobian']=np.array(matrices,np.float32)
    np.savez_compressed(args.cache/'cache.npz',**values)
    metadata=json.loads((args.cache/'cache.json').read_text())
    metadata.update(position_names=schema['state_joint_names'],velocity_names=schema['action_actuator_names'],
        proposal_names=schema['action_actuator_names'],output_arm_actuator_names=schema['action_actuator_names'][:7],
        training_loss='Linearized own-wrist translation error at 3 mm scale, plus joint-error and small-residual regularization')
    (args.cache/'cache.json').write_text(json.dumps(metadata,indent=2))
    print('Added robot kinematic Jacobians; no object poses or source timing used.')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('runs/book5-g1-ego-09'));main(p.parse_args())
