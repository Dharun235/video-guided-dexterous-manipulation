import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from retargeting.model_bridge import export_robot


def fixture(poly='0 1.334 0 0 0'):
    sites = ''.join(f'<site name="landmark_{i}" pos="0 .01 0"/>' for i in range(21))
    return mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="pelvis" pos="0 0 .793"><body name="torso_link">
        <body name="left_arm"><joint name="left_elbow_joint"/>
          <geom type="sphere" size=".01"/></body>
        <body name="right_hand"><joint name="driver" range="-90 90"/>
          <geom type="sphere" size=".01"/>{sites}
          <body name="finger" pos="0 .03 0"><joint name="dependent" range="-90 90"/>
            <geom type="sphere" size=".01"/></body>
        </body></body></body></worldbody>
      <equality><joint joint1="dependent" joint2="driver" polycoef="{poly}"/></equality>
      <actuator><position name="driver" joint="driver" ctrlrange="-1 1"/></actuator>
      </mujoco>''')


def test_export_preserves_mimic_and_fixes_unobserved_arm(tmp_path):
    path, translation = export_robot(fixture(), np.array([0]), tmp_path)
    root = ET.parse(path).getroot()
    assert root.find("joint[@name='dependent']/mimic").get('joint') == 'driver'
    assert float(root.find("joint[@name='dependent']/mimic").get('multiplier')) == pytest.approx(1.334)
    assert root.find("joint[@name='left_elbow_joint']") is None
    assert root.find("joint[@name='left_arm__fixed']").get('type') == 'fixed'
    np.testing.assert_allclose(translation, [0, 0, .793])
    assert len([n for n in root.findall('link') if n.get('name').startswith('landmark_')]) == 21


def test_export_rejects_lossy_nonlinear_coupling(tmp_path):
    with pytest.raises(ValueError, match='Nonlinear joint coupling'):
        export_robot(fixture('0 1 .1 0 0'), np.array([0]), tmp_path)
