"""Load pinned upstream code without copying it into this project's license."""
import importlib
import json
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = '238cd8b45f3969f24f9c66b91eb707ec1b51825a'


def load():
    path = ROOT / 'third_party/humanoid-dexart-retarget'
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Unexpected upstream revision')
    pins = json.loads((ROOT / 'retargeting/UPSTREAM.json').read_text())
    for name, expected in pins['source_files_sha256'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Upstream source differs from the inspected version: '+name)
    sys.path.insert(0, str(path))
    import jaxls
    if not hasattr(jaxls.Cost, 'create_factory'):
        jaxls.Cost.create_factory = jaxls.Cost.factory
    return importlib.import_module('retarget_h2o'), importlib.import_module('utils'), path


def default_weights():
    return dict(laplacian_deformation=2., joint_smoothness=1., root_smoothness=50.,
                posture_prior=5., root_upright=5., arm_prior=.2, self_collision=.5,
                floor_contact=5., foot_skating=20.)
