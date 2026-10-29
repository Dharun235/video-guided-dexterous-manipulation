import mujoco
import numpy as np

from humanoid.tactile import sense,GripFeedback


def model_for(body):
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
    <geom type="plane" size="1 1 .1"/>
    <body name="robot" pos="0 0 .015"><freejoint/>
    <body name="{body}"><geom type="sphere" size=".02" mass="1"/></body>
    </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return model,data


def test_external_fingertip_sensor_does_not_require_object_identity():
    model,data=model_for('R_thumb_distal')
    assert data.ncon>0
    values=sense(model,data)
    assert values.shape==(10,) and values[0]>0
    assert np.array_equal(values[2:],np.zeros(8))


def test_proximal_collision_is_not_a_fingertip_pad_reading():
    model,data=model_for('R_thumb_proximal')
    assert data.ncon>0
    assert np.array_equal(sense(model,data),np.zeros(10))


def test_contact_gate_requires_sustained_thumb_and_opposition(monkeypatch):
    signal=np.zeros(10)
    monkeypatch.setattr('humanoid.tactile.sense',lambda model,data:signal.copy())
    controller=GripFeedback();base=np.ones(13)*.5
    assert not controller(4.81,base,None,None)[1]
    signal[0]=2;signal[2]=2
    assert not controller(4.81,base,None,None)[1]
    assert not controller(4.81,base,None,None)[1]
    assert controller(4.81,base,None,None)[1]
    signal[0]=50
    for _ in range(50):controller(5.,base,None,None)
    assert -.1200001<=controller.trim[4]<0
