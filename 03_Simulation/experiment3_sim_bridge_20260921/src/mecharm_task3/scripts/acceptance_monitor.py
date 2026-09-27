#!/usr/bin/env python3
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from gazebo_msgs.srv import GetEntityState
from common import arguments,load_config,JsonLog
from validate_affine_sim import ARM_JOINTS

class Monitor(Node):
    def __init__(self,args):
        super().__init__('task3_acceptance_monitor');self.args=args;self.config=load_config(args.config)
        self.folder=Path(args.run_dir);self.log=JsonLog(args.run_dir,'safety.jsonl')
        self.joint_samples=0;self.joint_violations=[];self.physics=None;self.last_sample=0
        self.truth=json.loads((self.folder/'ground_truth.json').read_text());self.finished=False;self.started=time.monotonic()
        self.stop_pub=self.create_publisher(String,'/task3/emergency_stop',10);self.stop_sent=False
        self.create_subscription(JointState,'/joint_states',self.on_joint,10)
        self.create_subscription(String,'/task3/physics_safety',self.on_physics,10)
        self.client=self.create_client(GetEntityState,'/gazebo/get_entity_state')
        self.task=None;self.pending=None;self.index=0;self.positions={};self.create_timer(0.5,self.tick)

    def on_joint(self,msg):
        values=dict(zip(msg.name,msg.position));self.joint_samples+=1
        for name,(low,high) in zip(ARM_JOINTS,self.config['motion']['joint_limits']):
            value=values.get(name)
            if value is not None and not low-0.015<=value<=high+0.015:
                row={'joint':name,'position':value};self.joint_violations.append(row);self.log.write('joint_limit_violation',**row)
                self.emergency('joint_limit_violation')
        if time.monotonic()-self.last_sample>0.5:
            self.log.write('joint_sample',positions={k:values[k] for k in ARM_JOINTS if k in values});self.last_sample=time.monotonic()

    def on_physics(self,msg):
        self.physics=json.loads(msg.data)
        self.log.write('physical_contact_monitor',**self.physics)
        if self.physics['unsafe_contact_pairs']>0:self.emergency('unexpected_robot_contact')

    def emergency(self,reason):
        if not self.stop_sent:
            self.stop_pub.publish(String(data=json.dumps({'reason':reason})));self.stop_sent=True

    def tick(self):
        if self.finished:return
        taskpath=self.folder/'task_result.json'
        if self.task is None:
            if not taskpath.exists():return
            self.task=json.loads(taskpath.read_text())
        if self.pending:
            if not self.pending.done():return
            result=self.pending.result();key=self.truth['objects'][self.index]['name']
            self.positions[key]=None if result is None or not result.success else [result.state.pose.position.x,result.state.pose.position.y,result.state.pose.position.z]
            self.pending=None;self.index+=1
        if self.index<len(self.truth['objects']):
            req=GetEntityState.Request();req.name=self.truth['objects'][self.index]['name'];req.reference_frame='world'
            self.pending=self.client.call_async(req);return
        self.evaluate();self.finished=True

    def evaluate(self):
        correct=[];details=[]
        for obj in self.truth['objects']:
            label=obj['class_id'];pos=self.positions[obj['name']];matched=False
            if label in self.config['bins'] and pos:
                matched=any(abs(pos[0]-x)<=0.022 and abs(pos[1]-y)<=0.022 and abs(pos[2]-0.02)<=0.008 for x,y in self.config['bins'][label]['slots'])
            if matched:correct.append(obj['name'])
            details.append({**obj,'final_position':pos,'correct_bin':matched})
        states=self.task['processed_grids'];normal=self.args.scenario=='normal'
        exception_kinds=sorted(set(v for v in states.values() if v in ['EMPTY','UNKNOWN','FAIL','REJECTED']))
        safe=self.joint_samples>0 and not self.joint_violations and self.physics is not None and self.physics['unsafe_contact_pairs']==0
        recovered=[job for job in self.task['jobs'] if not job.get('ok') and job.get('safe_to_continue')]
        unreachable_handled=any(job.get('fault_injection')=='unreachable' and job.get('error')=='unreachable_or_collision_preflight' for job in recovered)
        grasp_failure_handled=any(job.get('fault_injection')=='grasp_failure' and any(step['label']=='grasp_alignment' and not step['ok'] for step in job.get('steps_before_failure',[])) for job in recovered)
        bridge_correct=sum(o['class_id']=='orange_bridge' and o['correct_bin'] for o in details)
        passed=(len(correct)>=self.config['task'].get('min_successes',5) and len(set(o['class_id'] for o in details if o['correct_bin']))>=2 and bridge_correct>=self.config['task'].get('required_bridge_successes',0)) if normal else ('EMPTY' in states.values() and 'UNKNOWN' in states.values() and unreachable_handled and grasp_failure_handled)
        result={'status':'PASS' if passed and safe and self.task['state']=='DONE' else 'FAIL',
                'scenario':self.args.scenario,'correct_count':len(correct),'tested_objects':sum(bool(x['class_id']) for x in details),
                'bridge_correct_count':bridge_correct,
                'safe':safe,'joint_samples':self.joint_samples,'joint_limit_violations':self.joint_violations,
                'physics_monitor':self.physics,'exception_kinds':exception_kinds,'recovered_failure_count':len(recovered),
                'unreachable_handled':unreachable_handled,'grasp_failure_handled':grasp_failure_handled,
                'details':details,'duration_wall_s':round(time.monotonic()-self.started,2),
                'evaluation_independent_of_detector':True,'task_state':self.task['state']}
        (self.folder/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('TASK3_ACCEPTANCE '+json.dumps(result),flush=True)

def main():
    args,rosargs=arguments();rclpy.init(args=rosargs);node=Monitor(args)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
