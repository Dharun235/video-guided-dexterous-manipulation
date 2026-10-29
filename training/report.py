"""Write a readable report from completed, locked learned-policy rollouts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from humanoid.book_dataset_audit import raised_hold_duration


def report(args):
    summary=json.loads((args.evaluation/'summary.json').read_text())
    if summary['scenarios_completed']!=summary['scenarios_expected']:raise ValueError('Evaluation is incomplete')
    rows=summary['reports']
    # Score all saved trajectories with the same corrected anatomical-finger
    # metric, including workers started before the counting fix.
    for i,r in enumerate(rows):
        p=args.evaluation/f'scenario_{i:02d}'
        x=np.load(p/'episode.npz');contacts=json.loads((p/'contacts.json').read_text())
        r['previous_reported_hold_s']=r['raised_multifinger_hold_s']
        r['raised_multifinger_hold_s']=raised_hold_duration(x['object_qpos'],contacts,20)
        r['checks']['sustained_raised_multifinger_hold']=r['raised_multifinger_hold_s']>=.5
        r['task_success']=all(r['checks'].values());r['grasp_metric_version']='distinct-anatomical-fingers-v2'
        (p/'report.json').write_text(json.dumps(r,indent=2))
    summary['task_successes']=sum(r['task_success'] for r in rows)
    for split,value in summary['per_suite'].items():
        selected_rows=[r for r in rows if r['evaluation_suite']==split]
        value['successes']=sum(r['task_success'] for r in selected_rows)
        value['check_pass_counts']={k:sum(r['checks'][k] for r in selected_rows) for k in selected_rows[0]['checks']}
    summary['grasp_metric_version']='distinct-anatomical-fingers-v2'
    (args.evaluation/'summary.json').write_text(json.dumps(summary,indent=2))
    locked=json.loads((args.dataset/'evaluation-scenarios.json').read_text())['scenarios']
    for split in ('validation','test'):
        expected={s['scene_xml'] for s in locked[split]}
        seen={r['scenario']['scene_xml'] for r in rows if r['evaluation_suite']==split}
        if seen!=expected:raise ValueError('Missing or extra locked '+split+' conditions')
    if any(r['teacher_enabled'] or any(r['physics'].values()) for r in rows):
        raise ValueError('Unexpected assistance or physics flag')
    training=json.loads((args.run/'run.json').read_text())
    checkpoint=Path(summary['checkpoint']);selected=json.loads((checkpoint/'training.json').read_text())
    original=next(r for r in rows if r['evaluation_suite']=='original')
    lines=['# Learned-policy results','',
        f"Trained SmolVLA checkpoint: {checkpoint}. Selected refinement step: {selected['steps_completed']}.",
        '', '| Evaluation | Successful / attempted | Mean planar error |',
        '|---|---:|---:|']
    for split,value in summary['per_suite'].items():
        lines.append(f"| {split} | {value['successes']} / {value['scenarios']} | {100*value['mean_planar_error_m']:.2f} cm |")
    lines+=['',f"Original-scene lift: {100*original['max_lift_m']:.2f} cm. "
        f"Placement error: {100*original['final_position_error_m']:.2f} cm. "
        f"Raised multi-finger hold: {original['raised_multifinger_hold_s']:.2f} s.",
        '', 'These are actual model-controlled physics rollouts. The teacher is disabled. '
        'The object remains free under gravity/contact; no attachment, object-pose playback or external object forces drive it.',
        '', 'The original scene is explicitly included as a training episode in the refinement. '
        'Its comparison is reproduction, not independent generalization. All movements derive from one human recording. '
        'Held-out appearance/physics conditions are narrow and correlated; condition counts are not independent human demonstrations.',
        '',f"The refinement warm-started from {training.get('initial_checkpoint')}, used "
        f"{training['steps_completed']} additional gradient steps, action horizon {training.get('chunk_size',50)}, "
        f"and nominal-episode sampling probability {training.get('extra_episode_sampling_probability',0)}. "
        'Normalization remains frozen from training data only. The test set did not select the checkpoint.',
        '', 'The initial 1,000-step run touched the case but failed the grasp. Its random and deterministic original-scene attempts remain saved. '
        'A lower offline loss must not be labelled physical task success.',
        '', 'Read summary.json for every condition and failure check, conditions.csv for a compact table, '
        'and original-comparison.mp4 for human/source and learned robot views. The human clip ends and holds independently of policy timing.',
        '', 'Next improvements should target the observed failures: corrective demonstrations at policy-visited states, '
        'movement/phase-balanced training, tighter visual workspace crops and validated control timing. '
        'Residual RL is a separate possible experiment: bounded corrections with lift/placement/release rewards and imitation regularization. '
        'No RL or hardware success is claimed by this report.']
    (args.evaluation/'README.md').write_text('\n'.join(lines)+'\n')
    with (args.evaluation/'conditions.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['suite','scene','physical_group','teacher_success','policy_success',
            'lift_cm','placement_error_cm','hold_seconds','penetration_mm','failed_checks'])
        for r in rows:
            s=r['scenario'];writer.writerow([r['evaluation_suite'],s['scene_xml'],s.get('physical_group'),
                s.get('teacher_task_success'),r['task_success'],100*r['max_lift_m'],100*r['final_position_error_m'],
                r['raised_multifinger_hold_s'],1000*r['maximum_penetration_m'],
                ','.join(k for k,v in r['checks'].items() if not v)])
    cap=cv2.VideoCapture(str(args.evaluation/'original-comparison.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));panels=[]
    for i in np.linspace(0,count-1,4,dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,frame=cap.read()
        if ok:panels.append(frame)
    cap.release()
    if len(panels)==4:cv2.imwrite(str(args.evaluation/'review-sheet.jpg'),np.vstack(panels))
    training.update(learned_policy_task_success_measured=True,evaluation=str(args.evaluation.resolve()),
        evaluation_results=summary['per_suite'])
    (args.run/'run.json').write_text(json.dumps(training,indent=2))
    shutil.copy2(args.evaluation/'summary.json',checkpoint/'EVALUATION.json')
    with (checkpoint/'README.md').open('a') as stream:
        stream.write('\nActual learned-policy rollout results:\n\n')
        for split,v in summary['per_suite'].items():stream.write(f"- {split}: {v['successes']}/{v['scenarios']} successful.\n")
        stream.write('\nSee EVALUATION.json for full conditions and failed checks.\n')
    files={str(p.relative_to(checkpoint)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(checkpoint.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (checkpoint/'manifest.json').write_text(json.dumps({'files_sha256':files},indent=2))
    print(json.dumps(summary['per_suite'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    report(p.parse_args())
