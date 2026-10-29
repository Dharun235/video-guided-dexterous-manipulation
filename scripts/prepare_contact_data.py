"""Finish a frozen preflight, render its verified data, then seal the package."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    while True:
        index=json.loads((args.dataset/'index.json').read_text());status=index['collection_status']
        if status=='preflight_complete':break
        if status.startswith('stopped'):raise RuntimeError('Preflight stopped; inspect retained failures: '+status)
        if status=='complete':break
        if not args.wait:raise ValueError('Preflight is still running; use --wait')
        time.sleep(10)
    locked=json.loads((args.dataset/'teacher-source/manifest.json').read_text())['files_sha256']
    for name,digest in locked.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen teacher changed: '+name)
    plan=json.loads((args.dataset/'plan.json').read_text())
    if status!='complete':
        command=[sys.executable,'-u','-m','humanoid.contact_collection','--output',str(args.dataset),
            '--groups',str(len(plan['groups'])),'--variants',str(plan['variants']),
            '--max-candidates','32','--resume']
        subprocess.run(command,check=True)
    subprocess.run([sys.executable,'-u','-m','humanoid.contact_finalize',str(args.dataset)],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--wait',action='store_true');main(parser.parse_args())
