"""Verify a transferred sealed dataset without importing the simulator."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    preparation=json.loads((root/'preparation.json').read_text())
    audit=json.loads((root/'collection-audit.json').read_text())
    if not preparation.get('data_ready') or not preparation.get('tactile_data_ready') or not audit['passed']:
        raise ValueError('Collection is not sealed/ready')
    manifest=json.loads((root/'manifest.json').read_text());bad=[]
    for name,expected in manifest['files_sha256'].items():
        p=root/name
        if not p.is_file():bad.append(name);continue
        with p.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:bad.append(name)
    if bad:raise ValueError('Missing or changed files: '+str(bad[:20]))
    print(json.dumps(dict(verified=True,files=len(manifest['files_sha256']),
        frames=preparation['frames'],tactile_state_dimension=preparation['tactile_input_dimension'],
        manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);verify(p.parse_args().folder)
