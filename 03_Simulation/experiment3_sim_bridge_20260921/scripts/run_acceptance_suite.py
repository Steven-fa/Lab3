#!/usr/bin/env python3
"""Run both real Gazebo scenarios, await evidence, stop only owned launch groups."""
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

bundle=Path(__file__).resolve().parents[1]
folder=bundle/'results'/('suite_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
folder.mkdir(parents=True,exist_ok=False)
summary=[]
for scenario in ['normal','exceptions']:
    run=folder/scenario;run.mkdir()
    env=os.environ.copy();env['TASK3_RUN_DIR']=str(run)
    with (run/'launch.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(['bash',str(bundle/'02_run_demo.sh'),scenario],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(f'Running {scenario}; evidence: {run}',flush=True)
    try:
        deadline=time.monotonic()+1200
        while not (run/'acceptance.json').exists():
            if process.poll() is not None:raise RuntimeError(f'{scenario}: launch exited before acceptance')
            if time.monotonic()>deadline:raise TimeoutError(f'{scenario}: acceptance timeout')
            time.sleep(2)
        result=json.loads((run/'acceptance.json').read_text(encoding='utf-8'))
        summary.append({'scenario':scenario,'status':result['status'],'correct_count':result['correct_count'],'path':str(run)})
        print(json.dumps(summary[-1]),flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
    time.sleep(3)
(folder/'suite_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
sys.exit(0 if len(summary)==2 and all(r['status']=='PASS' for r in summary) else 1)
