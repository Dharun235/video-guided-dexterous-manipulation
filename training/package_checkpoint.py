"""Save native LeRobot processors and provenance with the trained checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from .dataset import RobotDataset
from .smolvla import Adapter


def package(folder):
    config=SmolVLAConfig.from_pretrained(str(folder));config.device='cpu'
    config.vlm_model_name=json.loads(Path('training/UPSTREAM.json').read_text())['backbone']['repo_id']
    statistics=json.loads((folder/'statistics.json').read_text())
    stats={k:{name:torch.tensor(statistics[k][name],dtype=torch.float32)
              for name in ('mean','std','min','max')} for k in ('observation.state','action')}
    pre,post=make_smolvla_pre_post_processors(config,stats)
    metadata=json.loads((folder/'training.json').read_text())
    sample=RobotDataset(metadata['dataset'],'validation',include_tactile=metadata.get('tactile_inputs',False))[0]
    tokenizer=AutoProcessor.from_pretrained(folder/'backbone').tokenizer
    fake=SimpleNamespace(config=config,model=SimpleNamespace(vlm_with_expert=SimpleNamespace(
        processor=SimpleNamespace(tokenizer=tokenizer))))
    adapter=Adapter(fake,statistics,'cpu');custom=adapter.batch([sample],training=False)
    raw={k:torch.from_numpy(sample[k]) for k in custom if k in sample}
    raw['task']=sample['task'];native=pre(raw)
    errors={k:float(torch.max(torch.abs(native[k].float()-custom[k].float()))) for k in custom}
    if max(errors.values())>1e-5:raise ValueError('Native/custom preprocessing differs: '+str(errors))
    normalized=torch.zeros((1,50,13));assert torch.allclose(post(normalized),adapter.decode(normalized),atol=1e-6)
    (folder/'processor-verification.json').write_text(json.dumps(dict(passed=True,maximum_input_errors=errors,
        action_decode_matches=True),indent=2))
    pre.save_pretrained(folder,config_filename='policy_preprocessor.json')
    post.save_pretrained(folder,config_filename='policy_postprocessor.json')
    for name in ('UPSTREAM.json','requirements-smolvla-mac.lock.txt'):
        shutil.copy2(Path('training')/name,folder/name)
    shutil.copy2('training/APACHE-2.0.txt',folder/'APACHE-2.0.txt')
    shutil.copy2('.cache/smolvla-base/README.md',folder/'BASE_MODEL_CARD.md')
    state_description='nineteen joint positions plus ten external fingertip normal/shear forces' if metadata.get('tactile_inputs',False) else 'nineteen joint positions'
    (folder/'README.md').write_text(f'''# G1 / Inspire SmolVLA fine-tune

Local task-specific fine-tune of lerobot/smolvla_base, not a general-purpose
humanoid or walking policy. The pelvis is supported and fixed.

{metadata['steps_completed']} gradient steps; two RGB cameras, {state_description},
and thirteen absolute motor targets. The 256x256 images keep the
phone view letterboxed. Expert/state/action projections are trained; the visual
backbone is frozen. Stats use the training split only. All source motion derives
from one human recording. Test conditions were excluded from training/selection.

Run with the project interface (keeps MuJoCo in its separate environment):

    HF_HOME=.cache/huggingface .venv-smolvla/bin/python -m training.evaluate \\
      --checkpoint {folder} --replan-every {1 if metadata.get('tactile_inputs',False) else 5} --suite original --output runs/new-policy-original

Native LeRobot pre/post processors, normalization statistics, backbone
configuration/tokenizer and named feature configuration are included. The project
adapter also normalizes state/actions and uses the same language tokens. Send
decoded actions directly to the named actuators without teacher feedforward.

Weights have been trained, but training loss alone does not establish task
success. Read the separate actual rollout reports for grasp/lift/release and
placement results. The original-video comparison is reproduction: its motion
contributed to training. Use all locked held-out scenarios for generalization.

The base model and LeRobot use Apache-2.0. Attribution, revisions and dependencies
are in UPSTREAM.json and requirements-smolvla-mac.lock.txt. The local raw recording
and licensed MANO assets are not included in this checkpoint.
''')
    files={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (folder/'manifest.json').write_text(json.dumps({'files_sha256':files},indent=2))
    print(json.dumps(dict(checkpoint=str(folder.resolve()),files=len(files),native_processors_saved=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path)
    package(p.parse_args().folder)
