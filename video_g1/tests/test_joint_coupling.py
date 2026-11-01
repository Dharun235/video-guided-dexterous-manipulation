"""Ensure the native FK adapter uses model equations rather than reference-hand constants."""

import mujoco
import numpy as np
import pytest

from humanoid.kinematics import set_configuration


@pytest.mark.parametrize("offset,intermediate,distal", [(0.0, 1.6, 2.4), (0.15, 1.33, 0.66)])
def test_adapter_preserves_supplied_thumb_coupling(offset, intermediate, distal):
    model = mujoco.MjModel.from_xml_string(f'''
    <mujoco><worldbody><body><joint name="thumb_proximal_yaw_joint"/>
      <geom size=".01" mass=".01"/>
      <body><joint name="thumb_proximal_pitch_joint"/><geom size=".01" mass=".01"/>
        <body><joint name="thumb_intermediate_joint"/><geom size=".01" mass=".01"/>
          <body><joint name="thumb_distal_joint"/><geom size=".01" mass=".01"/></body>
        </body></body></body></worldbody>
    <equality>
      <joint joint1="thumb_intermediate_joint" joint2="thumb_proximal_pitch_joint" polycoef="{offset} {intermediate} 0 0 0"/>
      <joint joint1="thumb_distal_joint" joint2="thumb_proximal_pitch_joint" polycoef="{offset} {distal} 0 0 0"/>
    </equality><actuator><motor joint="thumb_proximal_yaw_joint"/><motor joint="thumb_proximal_pitch_joint"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    set_configuration(model, data, np.array([0.3, 0.2]))
    np.testing.assert_allclose(data.qpos, [0.3, 0.2, offset + 0.2 * intermediate, offset + 0.2 * distal])
