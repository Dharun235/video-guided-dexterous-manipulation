import mujoco

from humanoid.layout import verify_layout


def model(angle,y=0):
    return mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
      <worldbody><body name="pelvis"/><geom name="table_collision" type="box"
      size=".32 .14 .015" pos=".4 {y} .765" euler="0 0 {angle}"/></worldbody></mujoco>''')


def test_robot_faces_long_edge():
    assert verify_layout(model(1.5707963267948966))['passed']


def test_sideways_table_is_rejected():
    result=verify_layout(model(0))
    assert not result['passed']
    assert not result['checks']['long_edge_parallel_to_shoulders']


def test_robot_must_be_centred_on_edge():
    result=verify_layout(model(1.5707963267948966,.18))
    assert not result['checks']['robot_centred_on_long_edge']
