#!/usr/bin/env python3
"""Run this bundle in isolation and stop its launch after independent evaluation."""
import datetime
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

bundle=Path(__file__).resolve().parents[1]
probe='--probe' in sys.argv
folder=bundle/'results'/(('probe_' if probe else 'bridge_')+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
folder.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,SIM_GUI='false',TASK3_RUN_DIR=str(folder))
print('RUN_DIR '+str(folder),flush=True)
with (folder/'launch.log').open('w') as log:
    process=subprocess.Popen(['bash',str(bundle/'02_run_demo.sh'),'normal']+(['autostart:=false'] if probe else []),env=env,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    observer_log=(folder/'observer.log').open('w')
    observer_command='source '+shlex.quote(str(bundle/'scripts/env.sh'))+'; exec python3 '+shlex.quote(str(bundle/'scripts/record_side_view.py'))+' '+shlex.quote(str(folder))
    observer=subprocess.Popen(['bash','-lc',observer_command],env=env,stdout=observer_log,stderr=subprocess.STDOUT)
    started=time.monotonic()
    try:
        while process.poll() is None and time.monotonic()-started<1200:
            if probe and time.monotonic()-started>25:
                probe_command='source '+shlex.quote(str(bundle/'scripts/env.sh'))+'; exec python3 '+shlex.quote(str(bundle/'scripts/probe_reach.py'))+' '+shlex.quote(str(folder))
                with (folder/'reach_probe.jsonl').open('w') as output:
                    probe_result=subprocess.run(['bash','-lc',probe_command],env=env,stdout=output,stderr=subprocess.STDOUT,timeout=180)
                print('PROBE_EXIT '+str(probe_result.returncode),flush=True)
                break
            result=folder/'acceptance.json'
            if 'process has died' in (folder/'launch.log').read_text(errors='replace') and not result.exists():
                print('CHILD_PROCESS_DIED: see launch.log; run is not accepted',flush=True)
                break
            if result.exists():
                data=json.loads(result.read_text())
                print('ACCEPTANCE '+json.dumps(data),flush=True)
                # Allow terminal-state camera frame and video finalization.
                time.sleep(3)
                break
            time.sleep(2)
    finally:
        if observer.poll() is None:
            observer.send_signal(signal.SIGINT)
            try:observer.wait(timeout=10)
            except subprocess.TimeoutExpired:observer.terminate();observer.wait(timeout=5)
        observer_log.close()
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                process.wait(timeout=15)
    result=folder/'acceptance.json'
    if probe:sys.exit(probe_result.returncode if 'probe_result' in locals() else 2)
    if not result.exists():
        print('NO_ACCEPTANCE: inspect '+str(folder/'launch.log'),flush=True)
        sys.exit(2)
    sys.exit(0 if json.loads(result.read_text())['status']=='PASS' else 1)
