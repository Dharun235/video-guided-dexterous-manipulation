"""Package an audited pilot episode, with portable MuJoCo assets and JSONL rows."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def export(source,output):
    audit=json.loads((source/'audit.json').read_text())
    if not audit['passed']:raise ValueError('Only independently audited episodes may be exported')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('episode.npz','initial_state.npz','dataset.json','report.json','audit.json',
                 'contacts.json','retarget.json','retarget.npz','execution.mp4','review-sheet.jpg','layout-verification.json'):
        shutil.copy2(source/name,output/name)
    shutil.copytree(source/'images',output/'images')
    root=ET.parse(source/'scene.xml').getroot();(output/'assets').mkdir()
    for i,asset in enumerate(root.findall('asset/*')):
        if asset.get('file'):
            path=Path(asset.get('file'));destination=Path('assets')/f'{i:03d}_{path.name}'
            # MuJoCo derives unnamed mesh names from the file stem. Preserve that
            # name explicitly before renaming the copied file to avoid collisions.
            if asset.tag=='mesh' and not asset.get('name'):asset.set('name',path.stem)
            shutil.copy2(path,output/destination);asset.set('file',str(destination))
    (output/'scene.xml').write_text(ET.tostring(root,encoding='unicode'))
    schema=json.loads((output/'dataset.json').read_text())
    schema['task']='Pick up the black case from the book, move it to the right, and release it on the table.'
    schema['mujoco_version']=mujoco.__version__
    schema['features']={'observation.state':{'shape':[len(schema['state_joint_names'])],'dtype':'float64'},
                        'action':{'shape':[len(schema['action_actuator_names'])],'dtype':'float64'},
                        'observation.images.front':{'shape':[256,256,3],'dtype':'uint8'},
                        'observation.images.inspection':{'shape':[256,256,3],'dtype':'uint8'}}
    x=np.load(output/'episode.npz')
    with (output/'frames.jsonl').open('w') as stream:
        for i in range(len(x['timestamp'])):
            stream.write(json.dumps({'episode_index':0,'frame_index':i,'timestamp':float(x['timestamp'][i]),
                'task':schema['task'],'observation.state':x['observation_state'][i].tolist(),
                'action':x['action'][i].tolist(),
                'observation.images.front':f'images/observation.images.front/{i:06d}.jpg',
                'observation.images.inspection':f'images/observation.images.inspection/{i:06d}.jpg'})+'\n')
    (output/'dataset.json').write_text(json.dumps(schema,indent=2))
    repo=Path(__file__).resolve().parents[1]
    shutil.copy2(repo/'third_party/menagerie/unitree_g1/LICENSE',output/'G1_LICENSE.txt')
    shutil.copy2(repo/'third_party/unitree-xr-book/LICENSE',output/'INSPIRE_LICENSE.txt')
    shutil.copy2(repo/'humanoid/BOOK_G1_UPSTREAM.json',output/'ROBOT_UPSTREAM.json')
    (output/'README.md').write_text(f"""# G1/Inspire manipulation episode

{len(x['timestamp'])} synchronized observation/action pairs at {schema['fps']} Hz,
two 256x256 RGB cameras, {len(schema['state_joint_names'])} joint positions and
{len(schema['action_actuator_names'])} absolute motor targets.

The supported G1 faces the centre of the table's longer edge. The right hand is
an articulated five-finger Inspire model; the primary camera follows the robot's
viewpoint and preserves the source recording's aspect ratio without mirroring.
The task teacher adapts the recorded grasp to the robot. The case moves through
actuator-driven contact and gravity, with no object attachment or pose playback.

Images/state precede each action; next state follows 50 physics steps of 1 ms.
Read frames.jsonl for rows, episode.npz for exact arrays and dataset.json for
feature names and units. Object poses and source timestamps are privileged
verification fields, not policy observations.

The source episode passed physics, geometry, synchronization and replay checks.
These checks validate one simulated episode, not a learned policy or hardware.
Scene scale, hidden geometry and material parameters remain estimates. Training
has not started. This is a local JSONL/NPZ/JPEG package, not LeRobot parquet.

Robot source revisions and assumptions are in ROBOT_UPSTREAM.json. Model licenses
are included. Raw personal video and licensed human-hand assets are excluded.

Verify from the project root:

    python3 -m humanoid audit /absolute/path/to/this/folder --check-render
""")
    hashes={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(dict(source_episode=str(source.resolve()),
        selected_episodes=1,frames=len(x['timestamp']),training_scope='Audited single-instance pilot; no generalization claim',
        files_sha256=hashes),indent=2))
    print(f'Exported audited pilot to {output.resolve()}')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();export(a.source,a.output)
