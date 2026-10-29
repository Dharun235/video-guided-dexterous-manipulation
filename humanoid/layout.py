"""Verify that the G1 faces the centre of the table's longer edge."""
import json

import mujoco
import numpy as np


def verify_layout(model,output=None):
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    gid=model.geom('table_collision').id;bid=model.body('pelvis').id
    half=model.geom_size[gid,:2];rotation=data.geom_xmat[gid].reshape(3,3)
    longest=int(np.argmax(half));shortest=1-longest
    edge_axis=rotation[:,longest];normal=rotation[:,shortest]
    robot=data.xpos[bid];centre=data.geom_xpos[gid]
    body_rotation=data.xmat[bid].reshape(3,3);forward=body_rotation[:,0];shoulders=body_rotation[:,1]
    side=np.sign(np.dot(centre-robot,normal)) or 1
    near_edge=centre-side*half[shortest]*normal
    lateral=float(abs(np.dot(robot-near_edge,edge_axis)))
    heading=(centre-robot).copy();heading[2]=0;heading/=max(np.linalg.norm(heading),1e-9)
    checks=dict(horizontal_table=bool(abs(rotation[2,2])>.99),
        long_edge_parallel_to_shoulders=bool(abs(np.dot(edge_axis,shoulders))>.98),
        robot_facing_table=bool(np.dot(forward,heading)>.98),
        robot_centred_on_long_edge=bool(lateral<.05),
        robot_outside_table_footprint=bool(abs(np.dot(robot-centre,normal))>half[shortest]))
    result=dict(checks=checks,passed=all(checks.values()),
        long_edge_length_m=float(2*half[longest]),short_edge_length_m=float(2*half[shortest]),
        lateral_offset_from_edge_centre_m=lateral,nearest_long_edge_centre_m=near_edge.tolist(),
        robot_position_m=robot.tolist(),robot_forward_axis=forward.tolist())
    if output is not None:(output/'layout-verification.json').write_text(json.dumps(result,indent=2))
    return result
