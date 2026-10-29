"""Package the unchanged VLA and its learned feedback as a checksummed bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def package(args):
    selection=json.loads(args.selection.read_text())
    metadata=json.loads((args.feedback.parent/'metadata.json').read_text())
    if digest(args.feedback)!=selection['feedback_sha256']:
        raise ValueError('Selected feedback weights changed')
    if digest(args.checkpoint/'model.safetensors')!=metadata['base_weights_sha256']:
        raise ValueError('Base policy changed')
    args.output.mkdir(parents=True,exist_ok=False)
    # APFS copy-on-write avoids consuming another gigabyte on the training Mac.
    if sys.platform=='darwin':
        subprocess.run(['cp','-cR',str(args.checkpoint),str(args.output/'base')],check=True)
    else:shutil.copytree(args.checkpoint,args.output/'base')
    feedback=args.output/'feedback';feedback.mkdir()
    shutil.copy2(args.feedback,feedback/'feedback.npz')
    shutil.copy2(args.selection,feedback/'selected-controller.json')
    metadata['gating']='Selected configuration in selected-controller.json: 5 cm own-wrist rise, latched, 1 s ramp'
    metadata['selected_mode']=selection['mode'];metadata['selected_gain']=selection['gain']
    metadata['base_checkpoint']='base'
    (feedback/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (args.output/'README.md').write_text(f'''# SmolVLA with learned arm feedback

The VLA weights are the unchanged fine-tuned SmolVLA checkpoint. A separately
trained small network corrects seven arm targets using robot joint positions,
motor velocities and the VLA's proposals. Six finger targets remain unchanged.
Own-robot kinematics filter the correction to preserve first-order wrist rotation.
This is supervised feedback, not reinforcement learning.

Selected gain {selection['gain']}, mode {selection['mode']}, activation at
{selection['lift_gate_m']} m own-wrist rise, one-second ramp, 0.12 rad correction
bound. Settings were selected on two training scenes before fresh paired testing.
No object poses, target coordinates or source timestamps enter this controller.

Run from the project root, with reconstructed assets and both local environments:

```bash
HF_HOME=.cache/huggingface .venv-smolvla/bin/python -m training.evaluate \\
  --checkpoint {args.output}/base --feedback {args.output}/feedback/feedback.npz \\
  --feedback-gain {selection['gain']} --feedback-mode {selection['mode']} \\
  --feedback-lift-gate {selection['lift_gate_m']} --noise zero \\
  --suite original --output runs/new-feedback-original
```

The bundled base/EVALUATION.json describes the earlier VLA alone. Improved
controller results are reported separately in the feedback evaluation report.
The original scene contributed to training: its video is reproduction evidence.
All human motion comes from one recording. The pelvis is fixed and supported.
No real-time, real-hardware or broad generalization claim is made.

Base model attribution, license, named features, normalization and local backbone
configuration/tokenizer are in base/. Project inference code is required.
manifest.json binds all bundle files, including both sets of learned weights.
''')
    files={str(p.relative_to(args.output)):digest(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output/'manifest.json').write_text(json.dumps(dict(files_sha256=files),indent=2))
    if files['base/model.safetensors']!=metadata['base_weights_sha256']:raise ValueError('Copy mismatch')
    print(json.dumps(dict(bundle=str(args.output.resolve()),files=len(files),base_weights_unchanged=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('runs/book5-smolvla-02/best'))
    p.add_argument('--feedback',type=Path,default=Path('runs/book5-arm-feedback-01/model/feedback.npz'))
    p.add_argument('--selection',type=Path,default=Path('runs/book5-arm-feedback-01/selected-controller.json'))
    p.add_argument('--output',type=Path,required=True);package(p.parse_args())
