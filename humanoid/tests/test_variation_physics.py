"""Appearance randomization must not silently change nominal physics."""
import cv2
import mujoco
import numpy as np

from humanoid.book_variations import randomize_xml


def test_nominal_variation_preserves_material_friction(tmp_path):
    image=tmp_path/'base.png';cv2.imwrite(str(image),np.full((8,8,3),128,np.uint8))
    xml=f'''<mujoco><default><geom friction=".7 .005 .001"/></default>
    <asset><texture name="table" type="2d" file="{image}"/>
    <texture name="wall" type="2d" file="{image}"/></asset><worldbody>
    <camera name="front_dataset" pos="0 -1 1" quat="1 0 0 0" fovy="50"/>
    <camera name="inspection_dataset" pos="1 -1 1" quat="1 0 0 0" fovy="50"/>
    <geom name="table_collision" type="box" size=".3 .2 .01"/>
    <geom name="book_collision" type="box" size=".1 .06 .01" pos="0 0 .02"/>
    <body name="case" pos="0 0 .05"><freejoint/>
    <geom name="case_solid" type="box" size=".03 .02 .013" mass=".055"/></body>
    <body name="R_finger" pos=".3 0 .1"><geom type="sphere" size=".01" friction="1 .005 .001"/></body>
    </worldbody></mujoco>'''
    baseline=mujoco.MjModel.from_xml_string(xml)
    varied=randomize_xml(xml,tmp_path,123,'train',.055,1.,np.zeros(3))
    assert np.array_equal(baseline.geom_friction,varied.geom_friction)
    assert np.array_equal(baseline.body_mass,varied.body_mass)
    assert np.array_equal(baseline.body_inertia,varied.body_inertia)


def test_hold_requires_two_fingers_and_continuous_lift():
    from humanoid.book_dataset_audit import raised_hold_duration
    objects=np.zeros((5,7));objects[1:,2]=.04
    contacts=[{'contacts':[{'body':'R_thumb_distal','normal_force_n':1.},
                           {'body':'R_index_intermediate','normal_force_n':1.}]} for _ in range(5)]
    assert raised_hold_duration(objects,contacts,20)==.2
    contacts[2]['contacts'].pop()
    assert raised_hold_duration(objects,contacts,20)==.1


def test_two_segments_of_one_finger_do_not_count_as_two_fingers():
    from humanoid.book_dataset_audit import raised_hold_duration
    objects=np.zeros((12,7));objects[1:,2]=.04
    contacts=[{'contacts':[{'body':'R_thumb_distal','normal_force_n':1.},
                           {'body':'R_thumb_intermediate','normal_force_n':1.}]} for _ in range(12)]
    assert raised_hold_duration(objects,contacts,20)==0.
