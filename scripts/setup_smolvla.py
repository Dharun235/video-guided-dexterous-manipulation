"""Create the isolated Mac training environment and fetch pinned public weights."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    uv=shutil.which('uv') or '/opt/homebrew/bin/uv'
    python=ROOT/'.venv-smolvla/bin/python'
    env=dict(os.environ,UV_CACHE_DIR=str(ROOT/'.cache/uv'),HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_DISABLE_XET='1')
    if not python.exists():
        subprocess.run([uv,'venv','--python',str(ROOT/'.venv-g1/bin/python'),str(python.parent.parent)],check=True,env=env)
    subprocess.run([uv,'pip','install','--python',str(python),'-r',str(ROOT/'training/requirements-smolvla-mac.lock.txt')],check=True,env=env)
    script='''
import json
from pathlib import Path
from huggingface_hub import snapshot_download
p=json.loads(Path('training/UPSTREAM.json').read_text())
snapshot_download(p['repo_id'],revision=p['revision'],local_dir='.cache/smolvla-base',
    allow_patterns=['*.json','*.safetensors','README.md','LICENSE*'])
b=p['backbone']
snapshot_download(b['repo_id'],revision=b['revision'],local_dir='.cache/smolvla-base/backbone',
    allow_patterns=['config.json','*token*','*processor*','chat_template*','vocab*','merges.txt','added_tokens.json'])
print('Pinned SmolVLA weights and backbone processor are ready locally.')
'''
    subprocess.run([str(python),'-c',script],cwd=ROOT,check=True,env=env)


if __name__=='__main__':main()
