"""Render independent verified episodes; one master owns the collection index."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import contextlib
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import uuid

import mujoco
import numpy as np

from .book_variations import randomize_xml,_save_json
from .book_dataset_audit import audit
from .contact_collection import execute,quality


def render(job):
    root=Path(job['root']);group=job['group'];variant=job['variant'];chosen=job['chosen'];split=chosen['split']
    relative=f'{split}/group_{group:02d}/episode_{variant:02d}';folder=root/relative
    seed=400000+group*100+variant
    if folder.exists():folder.rename(folder.with_name(folder.name+'_interrupted_'+uuid.uuid4().hex[:6]))
    folder.mkdir();shutil.copy2(root/chosen['folder']/'retarget.npz',folder/'retarget.npz')
    source=json.loads(Path('runs/book5-g1-ego-09/retarget.json').read_text())
    _save_json(folder/'retarget.json',dict(source,physical_group=group,contact_feedback=True,selected_preflight=chosen['folder']))
    schema=json.loads(Path('runs/book5-g1-ego-09/dataset.json').read_text())
    instance=randomize_xml((root/'setup/scene.xml').read_text(),folder,seed,split,
        chosen['mass_kg'],chosen['friction_multiplier'],np.array(chosen['delta_m']))
    active=np.array(schema['active_actuator_indices']);state=np.array([instance.joint(n).id for n in schema['state_joint_names']])
    with (folder/'collection.log').open('w') as stream,contextlib.redirect_stdout(stream):
        execute(instance,active,state,folder,chosen,job['goal'],True)
        try:audit(folder,check_render=True)
        except SystemExit:pass
    verified=json.loads((folder/'audit.json').read_text())['passed'] and quality(folder)['passed']
    if not verified:raise RuntimeError('Recorded variant failed; excluded: '+relative)
    x=np.load(folder/'episode.npz')
    with (folder/'frames.jsonl').open('w') as stream:
        for i in range(len(x['action'])):
            row=dict(frame_index=i,timestamp=float(x['timestamp'][i]),task=schema['task'],**{
                'observation.state':x['observation_state'][i].tolist(),
                'observation.tactile':x['observation_tactile'][i].tolist(),'action':x['action'][i].tolist(),
                'observation.images.front':f'images/observation.images.front/{i:06d}.jpg',
                'observation.images.inspection':f'images/observation.images.inspection/{i:06d}.jpg'})
            stream.write(json.dumps(row)+'\n')
    return dict(episode=relative,physical_group=group,split=split,domain_seed=seed,frames=len(x['action']),
        task_success=True,verified=True,training_eligible=split=='train',preflight=chosen['folder'])


def run(args):
    root=args.dataset.resolve();index=json.loads((root/'index.json').read_text());plan=json.loads((root/'plan.json').read_text())
    locked=json.loads((root/'teacher-source/manifest.json').read_text())['files_sha256']
    for name,expected in locked.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('Teacher changed: '+name)
    completed={e['episode'] for e in index['episodes']};jobs=[]
    for group in plan['groups']:
        g=group['physical_group'];selection=root/group['split']/f'group_{g:02d}'/'selection.json'
        if not selection.exists():raise ValueError('Preflight incomplete')
        chosen=json.loads(selection.read_text())
        if not chosen['passed']:raise ValueError('Invalid selected teacher')
        for variant in range(plan['variants']):
            if f'{group["split"]}/group_{g:02d}/episode_{variant:02d}' not in completed:
                jobs.append(dict(root=str(root),group=g,variant=variant,chosen=chosen,goal=plan['fixed_destination_m']))
    index['collection_status']='rendering_parallel';_save_json(root/'index.json',index);failures=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(render,job):job for job in jobs}
        for future in as_completed(futures):
            try:
                entry=future.result();index['episodes'].append(entry)
                index['episodes'].sort(key=lambda e:(e['physical_group'],e['domain_seed']))
                _save_json(root/'index.json',index)
                print(f'Verified {len(index["episodes"])}/{index["groups"]*index["variants_per_group"]}: {entry["episode"]}',flush=True)
            except Exception as exc:
                failures.append(dict(job=futures[future],error=str(exc)))
                _save_json(root/'render-failures.json',failures);print(str(exc),flush=True)
    if failures:
        index['collection_status']='stopped_failed_parallel';_save_json(root/'index.json',index)
        raise RuntimeError('Recorded failures retained; collection is not ready')
    index['collection_status']='complete';_save_json(root/'index.json',index)
    if args.finalize:
        from .contact_finalize import seal
        seal(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--finalize',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=2:raise SystemExit('Use at most two render workers on this Mac')
    run(a)
