import numpy as np

from humanoid.arm_feedback import project_translation


def test_translation_projection_removes_rotation():
    jac=np.c_[np.eye(6),np.array([.2,.1,-.1,.1,.1,.2])]
    delta=np.array([.01,-.02,.03,.04,-.03,.02,.01])
    result=project_translation(delta,jac[:3],jac[3:])
    assert np.linalg.norm(jac[3:]@result)<1e-6
    assert np.allclose(jac[:3]@result,jac[:3]@delta,atol=1e-6)


def test_projection_caps_joint_changes_without_adding_rotation():
    jac=np.c_[np.eye(6),np.ones(6)*.1]
    result=project_translation(np.ones(7)*10,jac[:3],jac[3:])
    assert np.max(np.abs(result))<=.12000001
    assert np.linalg.norm(jac[3:]@result)<1e-6
