"""Check coupled native MuJoCo derivatives against an independent perturbation."""

import mujoco
import numpy as np

from humanoid.kinematics import set_configuration
from video_g1.geometry import coupled_control_map, site_jacobians


def test_native_joint_coupling_enters_site_derivative():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><joint name="driver" axis="0 0 1"/>
      <geom type="sphere" size=".01" mass=".1"/>
      <body pos=".1 0 0"><joint name="dependent" axis="0 1 0"/>
        <geom type="sphere" size=".01" mass=".1"/><site name="tip" pos=".1 .02 .03"/>
      </body></body></worldbody>
      <equality><joint joint1="dependent" joint2="driver" polycoef=".1 .7 0 0 0"/></equality>
      <actuator><position joint="driver"/></actuator></mujoco>""")
    data = mujoco.MjData(model)
    set_configuration(model, data, [0.4])
    analytic = site_jacobians(model, data, [model.site("tip").id], coupled_control_map(model, [0]))
    epsilon = 1e-6
    set_configuration(model, data, [0.4 + epsilon])
    positive = data.site_xpos.copy()
    set_configuration(model, data, [0.4 - epsilon])
    negative = data.site_xpos.copy()
    np.testing.assert_allclose(analytic[:, :, 0], (positive - negative) / (2 * epsilon), atol=1e-9)
