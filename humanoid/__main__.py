"""Run the focused G1 workflow in its isolated environment."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
COMMANDS={'retarget':'humanoid.retarget_hand','audit':'humanoid.book_dataset_audit','preview':'humanoid.book_ego_preview'}

def main():
    args=sys.argv[1:]
    if not args or args[0] in ('--help','-h'):
        print('Video-guided G1/Inspire manipulation\nCommands: '+', '.join(COMMANDS)+'\nSee docs/running.md for arguments.')
        return
    if args[0] not in COMMANDS:raise SystemExit('Unknown command; use python3 -m humanoid --help')
    python=ROOT/'.venv-g1/bin/python'
    if not python.exists():raise SystemExit('Run python3 -m scripts.setup_g1 first')
    os.chdir(ROOT)
    os.execv(str(python),[str(python),'-m',COMMANDS[args[0]],*args[1:]])

if __name__=='__main__':main()
