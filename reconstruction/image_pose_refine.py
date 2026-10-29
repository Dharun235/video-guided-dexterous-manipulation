"""Bounded image-space pose refinement, respecting estimated hand occlusion."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def refine_pose(surface, rotation, translation, camera_r, camera_t, k, case, occluder):
    rotation = Rotation.from_matrix(rotation).as_matrix()
    camera_points = (surface @ rotation.T + translation - camera_t) @ camera_r
    if camera_points[:, 2].min() <= .05:
        return None
    contours, _ = cv2.findContours(case, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    edge = np.concatenate(contours).reshape(-1, 2)
    near_hand = cv2.dilate(occluder, np.ones((7, 7), np.uint8)) > 0
    edge = edge[~near_hand[edge[:, 1], edge[:, 0]]]
    if len(edge) < 20:
        return None
    edge = edge[np.linspace(0, len(edge)-1, min(len(edge), 150), dtype=int)].astype(float)
    tree = cKDTree(edge)
    fy, fx = np.where(case > 0)
    foreground = np.c_[fx, fy].astype(float)
    foreground = foreground[np.linspace(0, len(foreground)-1, min(len(foreground), 120), dtype=int)]
    object_points = surface @ rotation.T @ camera_r
    centre = (translation - camera_t) @ camera_r

    def project(delta):
        points = object_points @ Rotation.from_rotvec(delta[:3]).as_matrix().T + centre + delta[3:]
        projected = points @ k.T
        uv = projected[:, :2] / projected[:, 2:]
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(float)
        closed = np.vstack([hull, hull[0]])
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
        samples = np.linspace(0, distances[-1], 64, endpoint=False)
        return np.c_[np.interp(samples, distances, closed[:, 0]), np.interp(samples, distances, closed[:, 1])]

    def objective(delta):
        uv = project(delta)
        # Observed case edges -> projected full-case boundary segments.
        a, b = uv, np.roll(uv, -1, axis=0)
        segment = b - a
        along = ((edge[:, None] - a[None]) * segment[None]).sum(axis=2)
        along /= np.maximum((segment ** 2).sum(axis=1), 1e-12)
        nearest = a[None] + along.clip(0, 1)[..., None] * segment[None]
        differences = nearest - edge[:, None]
        best = np.argmin((differences ** 2).sum(axis=2), axis=1)
        observed_error = differences[np.arange(len(edge)), best].ravel()
        # Model boundary -> observed edge; ignore boundary behind the hand.
        _, closest = tree.query(uv)
        pixels = np.rint(uv).astype(int)
        inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < case.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < case.shape[0])
        visible = inside.copy()
        visible[inside] &= ~near_hand[pixels[inside, 1], pixels[inside, 0]]
        model_error = (uv - edge[closest]) * visible[:, None]
        # Positive mask evidence also matters where a hand creates an internal
        # boundary: observed case pixels must lie inside the projected object.
        along_fg = ((foreground[:, None] - a[None]) * segment[None]).sum(axis=2)
        along_fg /= np.maximum((segment ** 2).sum(axis=1), 1e-12)
        closest_fg = a[None] + along_fg.clip(0, 1)[..., None] * segment[None]
        diff_fg = closest_fg - foreground[:, None]
        nearest_fg = np.argmin((diff_fg ** 2).sum(axis=2), axis=1)
        cross = segment[None, :, 0] * (foreground[:, None, 1] - a[None, :, 1]) - segment[None, :, 1] * (foreground[:, None, 0] - a[None, :, 0])
        inside_fg = (cross >= -1e-8).all(axis=1) | (cross <= 1e-8).all(axis=1)
        foreground_error = diff_fg[np.arange(len(foreground)), nearest_fg] * (~inside_fg)[:, None]
        prior = delta / np.array([.15, .15, .15, .01, .01, .005])
        return np.r_[observed_error, model_error.ravel(), foreground_error.ravel(), prior]

    def jacobian(parameters):
        steps = np.array([1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5])
        columns = []
        for i, step in enumerate(steps):
            plus, minus = parameters.copy(), parameters.copy()
            plus[i] += step
            minus[i] -= step
            columns.append((objective(plus) - objective(minus)) / (2 * step))
        return np.column_stack(columns)

    fit = least_squares(objective, np.zeros(6), jac=jacobian,
                        bounds=([-.35, -.35, -.35, -.02, -.02, -.025], [.35, .35, .35, .02, .02, .025]),
                        loss="soft_l1", f_scale=2., max_nfev=40)
    residual = objective(fit.x)[:len(edge)*2].reshape(-1, 2)
    median = float(np.median(np.linalg.norm(residual, axis=1)))
    if median > 3. or np.linalg.norm(fit.x[3:]) > .025 or np.linalg.norm(fit.x[:3]) > .5:
        return None
    refined_rotation = camera_r @ Rotation.from_rotvec(fit.x[:3]).as_matrix() @ camera_r.T @ rotation
    return dict(translation=translation + camera_r @ fit.x[3:], rotation=refined_rotation,
                camera_correction_m=fit.x[3:], rotation_correction_rad=fit.x[:3], median_visible_edge_error_px=median,
                visible_edge_samples=len(edge))


def relocalize(surface, rotation_prior, camera_r, camera_t, k, case, occluder, surface_depth):
    """Image/depth pose hypothesis; orientation prior is external and uncertain."""
    ys, xs = np.where(case > 0)
    if len(xs) < 100 or not np.isfinite(surface_depth):
        return None
    relative = surface @ rotation_prior.T @ camera_r
    z = float(surface_depth - np.quantile(relative[:, 2], .1))
    if not .1 < z < 1.5:
        return None
    centre = np.array([(np.median(xs)-k[0, 2])*z/k[0, 0], (np.median(ys)-k[1, 2])*z/k[1, 1], z])
    translation = camera_r @ centre + camera_t
    result = refine_pose(surface, rotation_prior, translation, camera_r, camera_t, k, case, occluder)
    if result is None or result["median_visible_edge_error_px"] > 2.:
        return None
    return result
