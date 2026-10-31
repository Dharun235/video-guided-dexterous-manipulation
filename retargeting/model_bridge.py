"""Export a kinematically equivalent, right-arm-only URDF from our compiled model."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from humanoid.book_g1 import BACKGROUND_CONTROL


def fmt(x):
    """Serialize numeric XML attributes with consistent precision."""
    return " ".join(f"{v:.12g}" for v in x)


def export_robot(model, active, output):
    """Export the supported robot to URDF with matching landmarks and joint coupling."""
    root = ET.Element("robot", name="supported_g1_inspire")
    pelvis = model.body("pelvis").id
    selected = {model.actuator_trnid[int(a), 0] for a in active}
    mimics = {}
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
            dependent, driver = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
            if np.any(model.eq_data[i, 2:5] != 0):
                raise ValueError("Nonlinear joint coupling cannot be represented by a URDF mimic")
            if driver in selected:
                mimics[dependent] = (driver, model.eq_data[i, 1], model.eq_data[i, 0])
    bodies = [pelvis]
    for b in range(pelvis + 1, model.nbody):
        if int(model.body_parentid[b]) in bodies:
            bodies.append(b)
    for b in bodies:
        name = model.body(b).name
        link = ET.SubElement(root, "link", name=name)
        # These conservative boxes are only inputs to the upstream capsule
        # self-collision penalty. MuJoCo retains its original collision meshes.
        for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            position = model.geom_pos[g].copy()
            size = model.geom_size[g].copy()
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                vertices = model.mesh_vert[
                    model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]
                ]
                low, high = vertices.min(0), vertices.max(0)
                position += Rotation.from_quat(model.geom_quat[g][[1, 2, 3, 0]]).apply((low + high) / 2)
                half = (high - low) / 2
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                half = size
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                half = np.full(3, size[0])
            elif model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                half = np.array([size[0], size[0], size[1] + size[0]])
            else:
                continue
            node = ET.SubElement(link, "collision")
            rotation = Rotation.from_quat(model.geom_quat[g][[1, 2, 3, 0]])
            ET.SubElement(node, "origin", xyz=fmt(position), rpy=fmt(rotation.as_euler("xyz")))
            ET.SubElement(ET.SubElement(node, "geometry"), "box", size=fmt(np.maximum(2 * half, 1e-5)))
        if b == pelvis:
            continue
        parent = model.body(int(model.body_parentid[b])).name
        rotation = Rotation.from_quat(model.body_quat[b][[1, 2, 3, 0]])
        position = model.body_pos[b].copy()
        if model.body_jntnum[b] > 1:
            raise ValueError("Multiple-joint body is not supported")
        if model.body_jntnum[b]:
            j = int(model.body_jntadr[b])
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Expected robot hinge joints")
            if model.qpos0[model.jnt_qposadr[j]] != 0:
                raise ValueError("Nonzero joint reference needs explicit conversion")
            axis, pivot = model.jnt_axis[j], model.jnt_pos[j]
            if j in selected or j in mimics:
                pivot_link = name + "__pivot"
                ET.SubElement(root, "link", name=pivot_link)
                node = ET.SubElement(root, "joint", name=model.joint(j).name, type="revolute")
                ET.SubElement(node, "parent", link=parent)
                ET.SubElement(node, "child", link=pivot_link)
                ET.SubElement(
                    node,
                    "origin",
                    xyz=fmt(position + rotation.apply(pivot)),
                    rpy=fmt(rotation.as_euler("xyz")),
                )
                ET.SubElement(node, "axis", xyz=fmt(axis))
                ET.SubElement(
                    node,
                    "limit",
                    lower=str(model.jnt_range[j, 0]),
                    upper=str(model.jnt_range[j, 1]),
                    effort="100",
                    velocity="10",
                )
                if j in mimics:
                    driver, multiplier, offset = mimics[j]
                    ET.SubElement(
                        node,
                        "mimic",
                        joint=model.joint(driver).name,
                        multiplier=str(multiplier),
                        offset=str(offset),
                    )
                fixed = ET.SubElement(root, "joint", name=name + "__offset", type="fixed")
                ET.SubElement(fixed, "parent", link=pivot_link)
                ET.SubElement(fixed, "child", link=name)
                ET.SubElement(fixed, "origin", xyz=fmt(-pivot), rpy="0 0 0")
                continue
            angle = BACKGROUND_CONTROL.get(model.joint(j).name, 0.0)
            extra = Rotation.from_rotvec(axis * angle)
            position += rotation.apply(pivot - extra.apply(pivot))
            rotation = rotation * extra
        node = ET.SubElement(root, "joint", name=name + "__fixed", type="fixed")
        ET.SubElement(node, "parent", link=parent)
        ET.SubElement(node, "child", link=name)
        ET.SubElement(node, "origin", xyz=fmt(position), rpy=fmt(rotation.as_euler("xyz")))
    for i in range(21):
        sid = model.site(f"landmark_{i}").id
        name = f"landmark_{i}"
        ET.SubElement(root, "link", name=name)
        node = ET.SubElement(root, "joint", name=name + "__site", type="fixed")
        ET.SubElement(node, "parent", link=model.body(int(model.site_bodyid[sid])).name)
        ET.SubElement(node, "child", link=name)
        ET.SubElement(node, "origin", xyz=fmt(model.site_pos[sid]), rpy="0 0 0")
    # The generic upstream solver indexes a head link, even when its upright
    # weight is zero. This fixed alias introduces no observed human head target.
    if not any(n.get("name") == "head_link" for n in root.findall("link")):
        ET.SubElement(root, "link", name="head_link")
        node = ET.SubElement(root, "joint", name="head_alias", type="fixed")
        ET.SubElement(node, "parent", link="torso_link")
        ET.SubElement(node, "child", link="head_link")
        ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
    path = Path(output) / "supported-g1-inspire.urdf"
    path.write_text(ET.tostring(root, encoding="unicode"))
    return path, model.body_pos[pelvis].copy()
