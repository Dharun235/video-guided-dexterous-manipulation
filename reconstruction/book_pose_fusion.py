"""Conservative, reference-assisted fusion of independent RGB tracking attempts.

This is not independent 6D recovery: the existing monocular reconstruction is
the depth/orientation prior and fallback. Every visual contribution is recorded.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from .book_space import case_vertices
from .book_motion import support_height, verify, render
from .book_visp import camera_at


def fuse(reference, attempts, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ('report.json', 'camera_poses.json', 'case_poses.json', 'hand_world.npz', 'hand_alignment.json'):
        shutil.copy2(reference/name, output/name)
    cases=json.loads((output/'case_poses.json').read_text())
    report=json.loads((output/'report.json').read_text())
    meta=json.loads((Path(report['scene'])/'reconstruction.json').read_text())
    local=case_vertices(meta['priors']['case_dimensions_m'])
    cameras=json.loads((output/'camera_poses.json').read_text())
    tracks=[(str(path.resolve()),json.loads((path/'poses.json').read_text())) for path in attempts]
    corrections=[]
    for row in cases:
        prior=np.array(row['position_m']); rotation=Rotation.from_quat(row['quaternion_xyzw'])
        candidates=[]
        # Static support intervals already have reviewed fits. Refine carrying only.
        if 5.05 < row['seconds'] < 9.8:
            for path, poses in tracks:
                candidate=min(poses,key=lambda p:abs(p['seconds']-row['seconds']))
                if not candidate['accepted_by_heuristic'] or (candidate['visible_mask_iou'] or 0)<.6:
                    continue
                matrix=np.array(candidate['object_to_camera'])
                rc,tc=camera_at(cameras,row['seconds'])
                prior_camera=rc@prior+tc
                # RGB tracking depth/rotation drifted: use its image center only,
                # explicitly retaining reference depth and orientation.
                corrected_camera=matrix[:3,3]*(prior_camera[2]/matrix[2,3])
                delta=rc.T@(corrected_camera-prior_camera)
                if np.linalg.norm(delta)>.015:
                    continue
                candidates.append((candidate['visible_mask_iou'],path,candidate,delta))
        evidence=dict(method='reference_fallback',reference_mode=row['mode'])
        if candidates:
            iou,path,candidate,delta=max(candidates,key=lambda x:x[0])
            # Limit influence to 3 mm; retain the reference orientation.
            alpha=.2
            position=prior+alpha*delta
            fitted=rotation
            points=local@fitted.as_matrix().T+position
            position[2]+=max(0.,float(np.max(support_height(points)-points[:,2])))
            row['position_m']=position.tolist();row['quaternion_xyzw']=fitted.as_quat().tolist()
            row['mode']='m3t_with_reference_prior'
            evidence=dict(method='m3t_with_reference_prior',reference_mode=evidence['reference_mode'],
                          tracking_attempt=path,source_frame=candidate['source_frame'],
                          visible_mask_iou=iou,visual_weight=alpha,
                          retained_from_reference=['camera_depth','orientation'],
                          translation_correction_m=float(np.linalg.norm(position-prior)),
                          rotation_correction_degrees=float(np.rad2deg((fitted*rotation.inv()).magnitude())))
        row['tracking_evidence']=evidence
        corrections.append(dict(seconds=row['seconds'],**evidence))
    report['fusion']={'reference':str(reference.resolve()),'tracking_attempts':[p for p,_ in tracks],
                      'refined_samples':sum(c['method']=='m3t_with_reference_prior' for c in corrections),
                      'note':'Reference-assisted visual refinement; fallback is the earlier hand-guided estimate. No independent pose ground truth.'}
    report['object_pose_modes']={m:sum(c['mode']==m for c in cases) for m in sorted({c['mode'] for c in cases})}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    (output/'case_poses.json').write_text(json.dumps(cases,indent=2))
    (output/'fusion_evidence.json').write_text(json.dumps(corrections,indent=2))
    verify(output)
    print(json.dumps(report['fusion'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--reference',type=Path,default=Path('runs/book5-motion-04'))
    p.add_argument('--attempts',type=Path,nargs='+',required=True)
    p.add_argument('--render',action='store_true')
    a=p.parse_args();fuse(a.reference,a.attempts,a.output)
    if a.render:render(a.output)
