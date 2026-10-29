"""Fetch pinned official Inspire hand assets without changing existing downloads."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    spec=json.loads((ROOT/'humanoid/BOOK_G1_UPSTREAM.json').read_text())['hand']
    path=ROOT/'third_party/unitree-xr-book'
    if not path.exists():
        subprocess.run(['git','clone','--filter=blob:none','--sparse','--no-checkout',spec['repository'],str(path)],check=True)
        subprocess.run(['git','-C',str(path),'sparse-checkout','set','--no-cone','/LICENSE','/assets/inspire_hand/inspire_hand_right.urdf','/assets/inspire_hand/meshes/*_R.STL','/assets/inspire_hand/meshes/R_hand_base_link.STL'],check=True)
        subprocess.run(['git','-C',str(path),'fetch','--depth','1','origin',spec['revision']],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',spec['revision']],check=True)
    revision=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if revision!=spec['revision']:raise SystemExit('Preserve the existing Inspire checkout before replacing its different revision')
    if not (path/spec['source']).exists():
        subprocess.run(['git','-C',str(path),'sparse-checkout','add','--no-cone','/assets/inspire_hand/inspire_hand_right.urdf','/assets/inspire_hand/meshes/*_R.STL','/assets/inspire_hand/meshes/R_hand_base_link.STL'],check=True)
    print('Pinned Inspire assets ready. Use the separate G1 runtime from python3 -m scripts.setup_g1.')


if __name__=='__main__':main()
