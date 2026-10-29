"""Install the pinned humanoid-dexart-retarget CPU environment separately."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    pins = json.loads((ROOT/'retargeting/UPSTREAM.json').read_text())
    target = ROOT/'third_party/humanoid-dexart-retarget'
    if not target.exists():
        subprocess.run(['git','clone','--filter=blob:none','--sparse','--no-checkout',pins['repository'],str(target)],check=True)
        subprocess.run(['git','-C',str(target),'sparse-checkout','set','--no-cone','/README.md','/retarget_h2o.py','/utils.py',
                        '/assets/g1_with_inspire_hand/*','/example_data/h2o/mano/subject1/o1/milk.mano.npy',
                        '/example_data/h2o/object_pose/subject1/o1/milk.object.npy','/example_data/h2o/object_mesh/milk/*'],check=True)
        subprocess.run(['git','-C',str(target),'fetch','--depth','1','origin',pins['revision']],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',pins['revision']],check=True)
    actual = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
    if actual != pins['revision']:
        raise SystemExit('Existing retargeter checkout has a different revision; preserve it before replacing')
    uv = shutil.which('uv')
    if not uv:
        raise SystemExit('Install uv first')
    env = dict(os.environ, UV_CACHE_DIR=str(ROOT/'.cache/uv'))
    python = ROOT/'.venv-dexart/bin/python'
    if not python.exists():
        subprocess.run([uv,'venv','--python','3.12',str(ROOT/'.venv-dexart')],check=True,env=env)
    subprocess.run([uv,'pip','install','--python',str(python),'-r',str(ROOT/'retargeting/requirements-mac.lock.txt')],check=True,env=env)
    subprocess.run([str(python),'-c','from retargeting.upstream import load; load(); import jax; print(jax.devices())'],cwd=ROOT,check=True)
    print('Pinned upstream retargeter ready in .venv-dexart')


if __name__ == '__main__':
    main()
