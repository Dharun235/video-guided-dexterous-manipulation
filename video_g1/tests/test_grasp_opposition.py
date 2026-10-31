"""Separate opposing normal forces from pressure on the same object face."""

import mujoco
import numpy as np

from manipulation.control import external_contacts
from video_g1.grasp_verify import ContactReadout, opposition


def fixture(top=False):
    """Create a synthetic contact fixture; this is not a successful robot grasp."""
    positions = ["-.034 0 .15", ".034 0 .15"] if not top else ["-.01 0 .174", ".01 0 .174"]
    return mujoco.MjModel.from_xml_string(f'''
      <mujoco><option gravity="0 0 0"/>
      <default><geom solref=".005 1"/></default>
      <worldbody><body name="pelvis">
        <body name="R_thumb_distal" pos="{positions[0]}"><geom type="sphere" size=".005"/></body>
        <body name="R_index_distal" pos="{positions[1]}"><geom type="sphere" size=".005"/></body>
      </body><body name="unknown_external_body" pos="0 0 .15"><freejoint/>
        <geom type="box" size=".03 .02 .02" mass=".055"/>
      </body></worldbody></mujoco>''')


def test_native_opposing_contacts_have_correct_force_signs():
    model = fixture()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    measured, vectors = ContactReadout(model).read(data)
    np.testing.assert_array_equal(measured, external_contacts(model, data))
    assert measured[0, 0] > 0.08 and measured[1, 0] > 0.08
    assert opposition(vectors[model.body("unknown_external_body").id]) > 0.99


def test_pressing_same_face_is_not_an_opposing_grasp():
    model = fixture(top=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    measured, vectors = ContactReadout(model).read(data)
    assert measured[0, 0] > 0.08 and measured[1, 0] > 0.08
    assert opposition(vectors[model.body("unknown_external_body").id]) == 0


def test_forces_on_different_bodies_do_not_form_opposition():
    thumb = np.zeros((5, 3))
    index = np.zeros((5, 3))
    thumb[0] = [1, 0, 0]
    index[1] = [-1, 0, 0]
    assert max(opposition(body) for body in [thumb, index]) == 0


def test_opposition_requires_configured_minimum_force():
    vectors = np.zeros((5, 3))
    vectors[0] = [0.2, 0, 0]
    vectors[1] = [-0.2, 0, 0]
    assert opposition(vectors, minimum_force=0.3) == 0
    assert opposition(vectors, minimum_force=0.08) == 1
