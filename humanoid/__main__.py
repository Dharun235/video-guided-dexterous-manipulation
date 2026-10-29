"""Run the focused G1 workflow in its isolated environment."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
COMMANDS={'retarget':('.venv-dexart','retargeting.video'),'baseline':('.venv-g1','humanoid.retarget_hand')}

def main():
    args=sys.argv[1:]
    if not args or args[0] in ('--help','-h'):
        print('Video-guided G1/Inspire manipulation\nCommands: '+', '.join(COMMANDS)+'\nSee docs/running.md for arguments.')
        return
    if args[0] not in COMMANDS:raise SystemExit('Unknown command; use python3 -m humanoid --help')
    environment,module=COMMANDS[args[0]]
    python=ROOT/environment/'bin/python'
    if not python.exists():raise SystemExit('Install the selected environment; see docs/retargeting.md')
    os.chdir(ROOT)
    os.execv(str(python),[str(python),'-m',module,*args[1:]])

if __name__=='__main__':main()
