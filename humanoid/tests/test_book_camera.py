"""Phone framing must preserve the handedness of the simulated image."""
import mujoco
import numpy as np

from humanoid.book_transfer import front_sensor_image


def test_phone_crop_preserves_right_side_and_pixel_geometry():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="source_match"/></worldbody></mujoco>')
    image=np.zeros((256,256,3),np.uint8)
    image[:,:,0]=np.arange(256,dtype=np.uint8)[None,:]
    image[:,:,1]=np.arange(256,dtype=np.uint8)[:,None]
    result=front_sensor_image(model,image)
    assert np.array_equal(result[128],image[128])
    assert result[128,230,0]==230 and result[128,20,0]==20
    assert not result[:56].any() and not result[199:].any()


def test_legacy_camera_data_is_unchanged():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="front_dataset"/></worldbody></mujoco>')
    image=np.random.default_rng(0).integers(0,256,(256,256,3),dtype=np.uint8)
    assert np.array_equal(front_sensor_image(model,image),image)
