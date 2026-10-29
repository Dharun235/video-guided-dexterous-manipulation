"""Report complete paired feedback tests without selecting on their outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def read(path):return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def media(baseline,original,output):
    old=cv2.VideoCapture(str(baseline/'original-comparison.mp4'))
    new=cv2.VideoCapture(str(original/'original-comparison.mp4'))
    fps=new.get(cv2.CAP_PROP_FPS)
    if not old.isOpened() or not new.isOpened() or fps<=0:raise ValueError('Missing comparison videos')
    same_fps=abs(old.get(cv2.CAP_PROP_FPS)-fps)<1e-6
    raw=output/'before-after-raw.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1024,256));snapshots=[];index=0
    while True:
        ok,after=new.read()
        if not ok:break
        if not same_fps:old.set(cv2.CAP_PROP_POS_MSEC,1000*index/fps)
        ok,before=old.read()
        if not ok:raise ValueError('Baseline video ends before improved video')
        panel=np.hstack([after[:,:256],before[:,256:512],after[:,256:512],after[:,768:1024]])
        for i,label in enumerate(('Source video','VLA baseline','VLA + arm feedback','Improved robot view')):
            cv2.rectangle(panel,(i*256,0),((i+1)*256,25),(0,0,0),-1)
            cv2.putText(panel,label,(i*256+5,17),0,.43,(255,255,255),1)
        writer.write(panel)
        if index in [round(t*fps) for t in (0,4,7,14)]:snapshots.append(panel.copy())
        index+=1
    writer.release();old.release();new.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(output/'before-after.mp4')],check=True)
    cv2.imwrite(str(output/'review.jpg'),np.vstack(snapshots))


def report(args):
    fresh=read(args.fresh/'summary.json');original=read(args.original/'summary.json');locked=read(args.locked/'summary.json')
    if fresh['scenarios_completed']!=14 or original['scenarios_completed']!=1 or locked['scenarios_completed']!=14:
        raise ValueError('All fresh pairs, locked conditions and original reproduction must finish')
    expected=read(args.dataset/'evaluation-scenarios.json')['scenarios']
    for split in ('validation','test'):
        seen=[r['scenario']['scene_xml'] for r in locked['reports'] if r['evaluation_suite']==split]
        if sorted(seen)!=sorted(s['scene_xml'] for s in expected[split]):raise ValueError('Locked conditions changed')
    selection=read(args.selection);plan=fresh['plan']
    if plan['feedback_sha256']!=selection['feedback_sha256']:raise ValueError('Feedback mismatch')
    for plan_key,key in (('feedback_gain','gain'),('feedback_mode','mode'),('feedback_lift_gate_m','lift_gate_m')):
        if plan[plan_key]!=selection[key]:raise ValueError('Selected controller configuration changed')
    if selection['fresh_test_results_used_for_selection']:raise ValueError('Test contamination')
    rows=fresh['reports'];new=original['reports'][0]
    previous=read(args.baseline/'summary.json')
    old=next(r for r in previous['reports'] if r['evaluation_suite']=='original')
    for r in rows+[new]+locked['reports']:
        if r['teacher_enabled'] or any(r['physics'].values()):raise ValueError('Unexpected assistance')
        if not r['arm_feedback']['finger_targets_unchanged']:raise ValueError('Modified finger commands')
        if r['task_success']!=all(r['checks'].values()):raise ValueError('Inconsistent task score')
        if r['arm_feedback']['enabled']:
            for key in ('gain','mode','lift_gate_m'):
                if r['arm_feedback'][key]!=selection[key]:raise ValueError('Runtime controller differs from selection')
    pairs=[]
    for i in range(7):
        b,f=rows[2*i:2*i+2]
        for key in ('scene_xml','initial_state_npz','schema','goal_position_m'):
            if b['scenario'][key]!=f['scenario'][key]:raise ValueError('Unpaired scene')
        pairs.append(dict(condition=i,baseline_success=b['task_success'],feedback_success=f['task_success'],
            baseline_error_cm=100*b['final_position_error_m'],feedback_error_cm=100*f['final_position_error_m'],
            baseline_failed_checks='; '.join(k for k,v in b['checks'].items() if not v),
            feedback_failed_checks='; '.join(k for k,v in f['checks'].items() if not v)))
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'paired-conditions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    with (args.output/'locked-conditions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['suite','episode','success','error_cm','failed_checks']);writer.writeheader()
        for r in locked['reports']:
            writer.writerow(dict(suite=r['evaluation_suite'],episode=r['scenario']['episode'],success=r['task_success'],
                error_cm=100*r['final_position_error_m'],failed_checks='; '.join(k for k,v in r['checks'].items() if not v)))
    groups=fresh['per_suite'];baseline=groups['fresh_baseline'];feedback=groups['fresh_feedback']
    new_cases=sum(not p['baseline_success'] and p['feedback_success'] for p in pairs)
    regressions=sum(p['baseline_success'] and not p['feedback_success'] for p in pairs)
    summary=dict(original_baseline=old,original_feedback=new,fresh_paired=groups,locked_feedback=locked['per_suite'],
        fresh_reports=rows,locked_reports=locked['reports'],locked_baseline=previous['per_suite'],
        deployment_recommendation='Retain VLA alone as the baseline; feedback is an experimental source-scene reproduction variant',
        additional_successes=new_cases,regressions=regressions,selection=selection,
        fresh_plan=plan,complete_pairs=7,grasp_metric_version='distinct-anatomical-fingers-v2',
        original_is_training_reproduction=True,reinforcement_learning_used=False,
        inference_source_sha256={str(p):digest(p) for p in map(Path,(
            'humanoid/arm_feedback.py','humanoid/policy_bridge.py','training/evaluate.py',
            'training/smolvla.py','training/paired_feedback.py','training/batch_feedback.py'))},
        evaluation_files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.fresh/'summary.json',args.original/'summary.json',args.locked/'summary.json',args.selection)})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2))
    media(args.baseline,args.original,args.output)
    lines=['# Learned arm-feedback results','',
        'The fine-tuned SmolVLA weights remain frozen. A new supervised arm-feedback network '
        'and robot-kinematic filtering correct placement. No reinforcement learning was used.','',
        '| Evaluation | VLA alone | VLA + feedback |', '|---|---:|---:|',
        f"| Original scene: placement error | {100*old['final_position_error_m']:.2f} cm | {100*new['final_position_error_m']:.2f} cm |",
        f"| Original scene: all physical task checks | {'pass' if old['task_success'] else 'fail'} | {'pass' if new['task_success'] else 'fail'} |",
        f"| Fresh paired conditions: all checks | {baseline['successes']} / 7 | {feedback['successes']} / 7 |",
        f"| Fresh paired conditions: mean placement error | {100*baseline['mean_planar_error_m']:.2f} cm | {100*feedback['mean_planar_error_m']:.2f} cm |",'',
        f"Original feedback rollout lifts {100*new['max_lift_m']:.2f} cm, sustains a distinct-finger raised hold "
        f"for {new['raised_multifinger_hold_s']:.2f} s, with maximum contact penetration "
        f"{1000*new['maximum_penetration_m']:.2f} mm. The placement threshold remains 3 cm.", '',
        f'Fresh tests add {new_cases} successful conditions and regress {regressions} previously successful conditions. '
        'Mean error includes failures; no conditions are dropped. See paired-conditions.csv for each failure.', '',
        'Visual inspection of failed fresh runs shows that the corrected controller can release the case '
        'onto the book or its edge instead of the tabletop target. Preserving grasp does not establish '
        'placement reliability. Additional corrective training trajectories are needed before treating '
        'this source-scene improvement as a generally stronger policy.', '',
        '## How the correction works','',
        'The network learns from 6,256 training examples (6,012 collection rows plus 244 nominal rows); '
        '1,428 validation examples select its training step. Inputs are 19 robot joint positions, '
        '13 motor velocities and 13 decoded VLA proposals. Outputs correct seven arm targets, bounded '
        'by 0.12 rad. Six finger commands remain the VLA outputs after native limit clipping. '
        'A Jacobian projection preserves first-order wrist rotation. The correction activates after '
        'the robot wrist rises 5 cm and ramps over one second. Its gain is 1.0. '
        'No object poses, target coordinates, contacts or source timestamps enter the controller.', '',
        'Gain/mode/gating were selected using two training scenes, before any fresh outcomes. '
        'Earlier raw corrections disturbed grasp formation and remain recorded. The unused grasp-offset '
        'servo prototype is archived separately and contributes nothing to these results.', '',
        '## Evaluation scope','',
        'Each fresh condition runs both controllers from the identical saved scene/reset/goal. '
        'The robot reset is fixed while the object moves up to 4 mm; mass ranges from 52 to 58 g, '
        'friction scales from 0.9 to 1.1, and appearances/cameras vary. No teacher filtering is applied. '
        'This differs from the older seven-condition benchmark, which adjusted robot initialization '
        'with each object spawn. Compare controllers within these paired tests; do not claim an '
        'unpaired improvement over the older 1/7 result.', '',
        'Both controllers use zero latent, five-step replanning and twenty seconds of simulation. '
        'Fresh inference batches two independent worlds; matching initial proposals are checked. '
        'Full-rate original reproduction uses batch one. MuJoCo runs at a 1 ms physics step, '
        '20 Hz control. Inference wall time is slower than simulation; real-time performance is not claimed.', '',
        'An earlier four-world locked-suite attempt was stopped because of memory pressure on the '
        '16 GB Mac. Its compact records are retained in runs/cleanup-records/source-and-old-run-records.tar.gz and excluded '
        'from complete-task counts. The completed replacement uses two worlds per inference batch '
        'with the identical controller and locked conditions.', '',
        'All motion derives from one recording. The original nominal scene was explicitly training data. '
        'Its improved video demonstrates reproduction, not an independent video test. Scene depth and '
        'hidden geometry remain estimates. The supported fixed-pelvis G1 does not balance or walk. '
        'These narrow synthetic tests cannot establish real-robot or broad task generalization.', '',
        'The next improvement should add corrective demonstrations from actual policy rollouts in '
        'training variations, including missed placements and recovery states. Fine-tune with a mixture '
        'of successful and corrective trajectories, then reserve new blind seeds for the next evaluation. '
        'The present fresh set has now been inspected and cannot remain a blind test for future changes.', '',
        '## Earlier locked benchmark','',
        'All fourteen original validation/test conditions, including teacher failures, were also run with '
        'the frozen feedback. They did not select the controller. Batched inference may introduce '
        'small numerical differences from the earlier batch-one baseline.','',
        '| Locked suite | Baseline successes | Feedback successes | Baseline mean error | Feedback mean error |',
        '|---|---:|---:|---:|---:|',
        *[f"| {split} | {previous['per_suite'][split]['successes']} / 7 | {v['successes']} / {v['scenarios']} | "
            f"{100*previous['per_suite'][split]['mean_planar_error_m']:.2f} cm | {100*v['mean_planar_error_m']:.2f} cm |"
            for split,v in locked['per_suite'].items()], '',
        'Keep the VLA alone as the baseline. The feedback bundle is an experimental source-scene '
        'reproduction variant, not a general-policy replacement.', '',
        '[Source / baseline / improved comparison video](before-after.mp4). [Review frames](review.jpg).',
        f'[Source and improved policy video](../{args.original.name}/original-comparison.mp4).',
        '[Every paired condition](paired-conditions.csv). [Every earlier locked condition](locked-conditions.csv). '
        '[Full measured results](results.json).', '',
        f'Policy bundle: `runs/book5-smolvla-feedback-01`. Base weights SHA-256: `{plan["base_weights_sha256"]}`. '
        f'Feedback SHA-256: `{plan["feedback_sha256"]}`. Reproduction commands: `training/README.md`.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    if args.bundle:
        manifest=read(args.bundle/'manifest.json')['files_sha256']
        if any(digest(args.bundle/name)!=value for name,value in manifest.items()):
            raise ValueError('Policy bundle was changed after packaging')
        (args.bundle/'EVALUATION.json').write_text(json.dumps(summary,indent=2))
        with (args.bundle/'README.md').open('a') as stream:
            stream.write('\n## Measured controller results\n\n'
                f"Original reproduction: {100*new['final_position_error_m']:.2f} cm placement error, "
                f"all checks {'pass' if new['task_success'] else 'fail'}. "
                f"Fresh paired conditions: baseline {baseline['successes']}/7, feedback {feedback['successes']}/7. "
                f"Mean error: {100*baseline['mean_planar_error_m']:.2f} cm baseline, "
                f"{100*feedback['mean_planar_error_m']:.2f} cm feedback. "
                'Read EVALUATION.json for all conditions, failures and original locked-suite results. '
                'The original scene was training data; source reproduction improvement does not establish broader reliability.\n')
        files={str(p.relative_to(args.bundle)):digest(p) for p in sorted(args.bundle.rglob('*'))
            if p.is_file() and p!=args.bundle/'manifest.json'}
        (args.bundle/'manifest.json').write_text(json.dumps(dict(files_sha256=files),indent=2))
    print(json.dumps(dict(original_error_cm=100*new['final_position_error_m'],fresh=groups),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    p.add_argument('--fresh',type=Path,default=Path('runs/book5-feedback-fresh-paired-01'))
    p.add_argument('--original',type=Path,default=Path('runs/book5-feedback-original-01'))
    p.add_argument('--baseline',type=Path,default=Path('runs/book5-smolvla-evaluation-02'))
    p.add_argument('--locked',type=Path,default=Path('runs/book5-feedback-locked-02'))
    p.add_argument('--selection',type=Path,default=Path('runs/book5-arm-feedback-01/selected-controller.json'))
    p.add_argument('--bundle',type=Path,default=Path('runs/book5-smolvla-feedback-01'))
    p.add_argument('--output',type=Path,required=True);report(p.parse_args())
