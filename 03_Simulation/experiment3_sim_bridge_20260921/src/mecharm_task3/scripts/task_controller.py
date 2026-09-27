#!/usr/bin/env python3
import json
from pathlib import Path
import time
import uuid
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from common import arguments,load_config,JsonLog,snapshot_grid

class TaskController(Node):
    def __init__(self,args):
        super().__init__('task3_controller');self.args=args;self.config=load_config(args.config)
        self.log=JsonLog(args.run_dir,'task_states.jsonl');self.state='WAIT_READY';self.ready=False
        self.grid=None;self.signature=None;self.stable=0;self.last_image=0.;self.boot=time.monotonic()
        self.processed={};self.jobs=[];self.pending=None;self.successes=0;self.bin_counts={k:0 for k in self.config['bins']}
        self.wait_until=0.;self.last_status=0.;self.finished=False
        self.requests=self.create_publisher(String,'/task3/grasp/request',10)
        self.cancel=self.create_publisher(String,'/task3/grasp/cancel',10)
        self.states=self.create_publisher(String,'/task3/state',10)
        self.create_subscription(Detection2DArray,'/task3/detections',self.on_detections,10)
        self.create_subscription(String,'/task3/grasp/status',self.on_arm_status,10)
        self.create_subscription(String,'/task3/grasp/result',self.on_result,10)
        self.create_subscription(String,'/task3/emergency_stop',lambda msg:self.finish('STOPPED',msg.data),10)
        self.create_timer(0.25,self.tick)

    def transition(self,state,detail=''):
        self.state=state;self.detail=detail
        row=self.log.write('state',state=state,detail=detail,successes=self.successes)
        self.states.publish(String(data=json.dumps(row)))

    def on_detections(self,msg):
        self.last_image=time.monotonic()
        self.grid=snapshot_grid(self.config,msg)
        signature=tuple((k,tuple((x['class_id'],x['score']>=self.config['detector']['min_confidence']) for x in v)) for k,v in self.grid.items())
        self.stable=self.stable+1 if signature==self.signature else 1;self.signature=signature

    def on_arm_status(self,msg):
        data=json.loads(msg.data);self.ready=data.get('ready',False)
        if self.pending and data.get('job_id')==self.pending['job_id']:
            self.log.write('grasp_feedback',**data)

    def on_result(self,msg):
        result=json.loads(msg.data)
        if not self.pending or result.get('job_id')!=self.pending['job_id']:
            self.log.write('ignored_unmatched_result',result=result);return
        self.jobs.append(result);grid_id=self.pending['grid_id'];self.processed[grid_id]=result.get('status','ERROR')
        self.log.write('job_result',**result)
        if result.get('ok'):
            self.successes+=1;self.bin_counts[self.pending['class_id']]+=1
            self.transition('VERIFY',f'G{grid_id}: placement verified')
        else:
            self.transition('RECOVER',result.get('error','motion failed'))
        self.pending=None;self.wait_until=time.monotonic()+self.config['task']['settle_after_job_s'];self.stable=0
        if result.get('safe_to_continue') is False:
            self.finish('STOPPED','Motion backend requested safe stop')

    def finish(self,state,reason):
        if self.finished:return
        self.finished=True;self.transition(state,reason)
        report={'state':state,'reason':reason,'scenario':self.args.scenario,'successes':self.successes,
                'processed_grids':self.processed,'bin_counts':self.bin_counts,'jobs':self.jobs,
                'selection_source':'vision_msgs/Detection2DArray from actual camera pixels',
                'manual_actions_after_launch':0}
        (Path(self.args.run_dir)/'task_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    def tick(self):
        now=time.monotonic()
        if now-self.last_status>1:
            self.states.publish(String(data=json.dumps({'state':self.state,'successes':self.successes,'detail':getattr(self,'detail','')})));self.last_status=now
        if self.finished:return
        if self.pending:
            if now-self.pending['issued_monotonic']>self.config['task']['job_timeout_s']:
                self.cancel.publish(String(data=json.dumps({'job_id':self.pending['job_id']})))
                self.log.write('job_timeout',job_id=self.pending['job_id'])
                self.finish('STOPPED','Long-running grasp exceeded deadline; cancellation sent')
            return
        if now<self.wait_until:return
        if not self.ready:
            if now-self.boot>90:self.finish('STOPPED','Arm backend not ready')
            return
        if self.last_image==0 or now-self.last_image>self.config['task']['image_timeout_s']:
            if now-self.boot>60:self.finish('STOPPED','Camera stream missing or stale')
            return
        if self.stable<self.config['task']['stable_observations']:return
        if len(self.processed)==len(self.config['grids']):
            self.finish('DONE','All grids processed or no executable objects remain');return
        self.transition('OBSERVE','Stable fresh image detections')
        for key in self.config['grids']:
            if key in self.processed:continue
            detections=self.grid[key]
            if not detections:
                self.processed[key]='EMPTY';self.log.write('empty_grid',grid_id=key);continue
            if len(detections)!=1:
                self.processed[key]='AMBIGUOUS';self.log.write('ambiguous_grid',grid_id=key,detections=detections);continue
            det=detections[0];label=det['class_id']
            if label not in self.config['bins'] or det['score']<self.config['detector']['min_confidence']:
                self.processed[key]='UNKNOWN';self.log.write('unknown_object',grid_id=key,detection=det);continue
            slot=self.bin_counts[label]
            if slot>=len(self.config['bins'][label]['slots']):
                self.processed[key]='BIN_FULL';self.log.write('bin_full',grid_id=key,class_id=label);continue
            job={'job_id':uuid.uuid4().hex,'grid_id':key,'class_id':label,'confidence':det['score'],
                 'bbox':det['bbox'],'bin_slot':slot,'issued_monotonic':now}
            self.pending=job;self.transition('SELECT',f"G{key} -> {self.config['bins'][label]['id']}[{slot}]")
            self.log.write('request',**job);self.requests.publish(String(data=json.dumps(job)))
            self.transition('EXECUTING',f'G{key} {label}');return

def main():
    args,rosargs=arguments();rclpy.init(args=rosargs);node=TaskController(args)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
