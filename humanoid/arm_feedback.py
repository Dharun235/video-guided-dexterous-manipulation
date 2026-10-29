"""Small learned arm-only feedback model; no object state or source timing."""
import numpy as np


def project_translation(delta,position_jacobian,rotation_jacobian,cap=.12):
    """Preserve first-order wrist orientation while keeping learned translation."""
    jac=np.vstack([position_jacobian,rotation_jacobian])
    twist=np.r_[position_jacobian@delta,np.zeros(3)]
    projected=jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(6),twist)
    return projected*min(1.,cap/max(float(np.max(np.abs(projected))),1e-8))


class ArmFeedback:
    def __init__(self,path,gain=1.):
        with np.load(path) as x:self.weights={k:x[k].copy() for k in x.files}
        if not 0<=gain<=2:raise ValueError('Feedback gain must be in [0,2]')
        self.gain=gain

    def correction(self,positions,velocities,proposal):
        feature=np.concatenate([positions,velocities,proposal]).astype(np.float32)
        if feature.shape!=(45,) or not np.isfinite(feature).all():raise ValueError('Invalid robot feedback inputs')
        w=self.weights;x=np.clip((feature-w['mean'])/w['std'],-8,8)
        for i in range(3):x=np.tanh(w[f'w{i}']@x+w[f'b{i}'])
        return np.clip(self.gain*w['cap']*x,-w['cap'],w['cap'])
