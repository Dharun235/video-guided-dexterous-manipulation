"""Choose arm-feedback gain using training-scene physical trials only."""
import argparse
import json
from pathlib import Path


def main(args):
    summary=json.loads((args.calibration/'summary.json').read_text())
    if summary['scenarios_completed']!=summary['scenarios_expected']:raise ValueError('Calibration incomplete')
    if not all(r['arm_feedback']['finger_targets_unchanged'] for r in summary['reports']):raise ValueError('Finger-target invariant failed')
    groups={}
    for r in summary['reports']:
        f=r['arm_feedback'];key=(f.get('mode','joint'),f.get('lift_gate_m',.025),f['gain'])
        groups.setdefault(key,[]).append(r)
    if not any(key[2]==0 for key in groups) or len(groups)<4 or any(len(rows)!=2 for rows in groups.values()):raise ValueError('Expected complete two-scene controller groups and baseline')
    scores=[]
    for (mode,gate,gain),rows in groups.items():
        preserved=sum(all(v for k,v in r['checks'].items() if k!='final_position_within_3cm') for r in rows)
        successes=sum(r['task_success'] for r in rows);error=sum(r['final_position_error_m'] for r in rows)/len(rows)
        scores.append(dict(gain=gain,mode=mode,lift_gate_m=gate,preserved_grasp_and_release=preserved,successes=successes,
            mean_planar_error_m=error,score=[preserved,successes,-error]))
    best=max(scores,key=lambda r:r['score']);baseline=next(r for r in scores if r['gain']==0)
    selected=dict(gain=best['gain'],improved_on_calibration=best['score']>baseline['score'],
        lift_gate_m=best['lift_gate_m'],mode=best['mode'],
        selected=best,baseline=baseline,candidates=scores,
        scope='Selection used original nominal training episode and one randomized training scene only. No fresh test outcomes used.')
    args.output.write_text(json.dumps(selected,indent=2));print(json.dumps(selected,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
