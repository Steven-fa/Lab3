#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from common import arguments,load_config,JsonLog

class GraspBridge(Node):
    def __init__(self,args):
        super().__init__('task3_grasp_bridge');self.args=args;self.config=load_config(args.config)
        self.log=JsonLog(args.run_dir,'grasp_interface.jsonl');self.folder=Path(args.run_dir)/'jobs';self.folder.mkdir(exist_ok=True)
        self.status=self.create_publisher(String,'/task3/grasp/status',10)
        self.results=self.create_publisher(String,'/task3/grasp/result',10)
        self.create_subscription(String,'/task3/grasp/request',self.request,10)
        self.create_subscription(String,'/task3/grasp/cancel',self.cancel,10)
        self.create_subscription(String,'/task3/emergency_stop',self.emergency,10)
        self.proc=None;self.job=None;self.ready=False;self.feedback='initializing';self.seen=set();self.started=time.monotonic()
        self.launch(None);self.create_timer(0.5,self.tick)

    def launch(self,job):
        key=job['job_id'] if job else 'prepare';out=self.folder/(key+'.json')
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('task3_motion.py')),'--config',self.args.config,'--run-dir',self.args.run_dir,'--output',str(out)]
        if job:
            req=self.folder/(key+'_request.json');req.write_text(json.dumps(job),encoding='utf-8');cmd+=['--request',str(req)]
            if self.args.scenario=='exceptions' and job['grid_id'] in ['5','6']:
                cmd+=['--fault',{'5':'unreachable','6':'grasp_failure'}[job['grid_id']]]
        else:cmd+=['--prepare-only']
        self.output=out;self.proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        self.job=job;self.started=time.monotonic()
        threading.Thread(target=self.read_output,args=(self.proc,key),daemon=True).start()

    def read_output(self,proc,key):
        with (self.folder/(key+'.log')).open('w',encoding='utf-8') as log:
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith('EVENT '):
                    event=json.loads(line[6:]);self.feedback=event.get('label','running')
                    self.log.write('feedback',job_id=key,step=event)

    def request(self,msg):
        try:
            job=json.loads(msg.data)
            if not re.fullmatch('[a-f0-9]{32}',str(job.get('job_id',''))):raise ValueError('invalid job_id')
            if job['grid_id'] not in self.config['grids']:raise ValueError('invalid grid')
            if job['class_id'] not in self.config['bins']:raise ValueError('unsupported class')
            if type(job['bin_slot']) is not int or not 0<=job['bin_slot']<len(self.config['bins'][job['class_id']]['slots']):raise ValueError('invalid bin slot')
            if job['job_id'] in self.seen:
                self.log.write('duplicate_ignored',job_id=job['job_id']);return
            if self.proc is not None or not self.ready:raise ValueError('busy_or_not_ready')
            self.seen.add(job['job_id']);self.log.write('accepted',**job);self.feedback='accepted';self.launch(job)
        except (ValueError,KeyError,TypeError) as exc:
            self.log.write('rejected',reason=str(exc))
            try:self.results.publish(String(data=json.dumps({'job_id':job.get('job_id'),'ok':False,'status':'REJECTED','error':str(exc),'safe_to_continue':True})))
            except Exception:pass

    def cancel(self,msg):
        request=json.loads(msg.data)
        if self.proc and self.job and request.get('job_id')==self.job['job_id']:
            self.log.write('cancel_requested',job_id=self.job['job_id']);os.killpg(self.proc.pid,signal.SIGINT)

    def tick(self):
        if self.proc and self.proc.poll() is not None:
            if self.output.exists():result=json.loads(self.output.read_text(encoding='utf-8'))
            else:result={'ok':False,'status':'FAIL','safe_to_continue':False,'error':'worker exited without result'}
            if self.job:
                result['job_id']=self.job['job_id'];self.results.publish(String(data=json.dumps(result)))
            else:self.ready=bool(result.get('ok'))
            self.log.write('worker_finished',result=result);self.proc=None;self.job=None;self.feedback='idle' if self.ready else 'not_ready'
        if self.proc and time.monotonic()-self.started>195:
            self.log.write('watchdog_timeout');os.killpg(self.proc.pid,signal.SIGINT);self.started=time.monotonic()
        self.status.publish(String(data=json.dumps({'ready':self.ready,'job_id':None if not self.job else self.job['job_id'],
                                                    'state':'BUSY' if self.proc else 'IDLE','feedback':self.feedback})))

    def emergency(self,msg):
        self.ready=False;self.log.write('emergency_stop',reason=msg.data)
        if self.proc and self.proc.poll() is None:os.killpg(self.proc.pid,signal.SIGINT)

    def close(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid,signal.SIGINT)
            try:self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:self.log.write('worker_not_stopped',pid=self.proc.pid)

def main():
    args,rosargs=arguments();rclpy.init(args=rosargs);node=GraspBridge(args)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.close();node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
