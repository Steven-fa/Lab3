#!/usr/bin/env python3
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import rclpy
from rclpy.signals import SignalHandlerOptions
from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import ApplyPlanningScene,GetPositionIK
from rcl_interfaces.msg import Parameter,ParameterValue
from rcl_interfaces.srv import SetParameters
from shape_msgs.msg import SolidPrimitive
import dynamic_affine_pick_place as d
from common import JsonLog,load_config

class Motion(d.DynamicAffinePickPlace):
    def __init__(self,config,folder,fault=''):
        super().__init__();self.config=config;self.fault=fault;self.held=False;self.prepared=None
        self.active_goal=None;self.events=JsonLog(folder,'motion_events.jsonl')
        self.selector=self.create_client(SetParameters,'/task3_grasp_plugin/set_parameters')
        self.step_records=[]
        d.APPROACH_HEIGHT_M=config['motion']['safe_object_z']-d.OBJECT_Z_M
        d.RELEASE_CLEARANCE_M=config['motion']['release_object_z']-d.OBJECT_Z_M

    def report_step(self,row):
        event=self.events.write('motion_step',target=self.target_name,**row)
        print('EVENT '+json.dumps(event),flush=True)

    def call(self,client,request,timeout=6):
        # These callers are reads or idempotent scene/parameter updates, never
        # motion goals or grasp triggers. Retry a lost discovery-time response.
        for attempt in range(3):
            if client.wait_for_service(timeout_sec=timeout):
                future=client.call_async(request)
                rclpy.spin_until_future_complete(self,future,timeout_sec=timeout)
                if future.done() and future.result() is not None:return future.result()
                future.cancel()
            self.events.write('service_retry',service=client.srv_name,attempt=attempt+1)
            rclpy.spin_once(self,timeout_sec=0.2)
        raise RuntimeError('service_timeout:'+client.srv_name)

    def position_of(self,name):
        pose=self.pose_of(name)
        if pose is None:return None
        p=pose.position;return [p.x,p.y,p.z]

    def pose_of(self,name,reference='world'):
        req=GetEntityState.Request();req.name=name;req.reference_frame=reference
        response=self.call(self.get_entity,req)
        if not response.success:return None
        return response.state.pose

    def collision_pose(self,attached=False):
        pose=self.pose_of(self.target_name,'mecharm_270_m5::link6' if attached else 'world')
        if pose is None:raise RuntimeError('collision_pose_unavailable')
        return pose

    def sync_world(self):
        req=ApplyPlanningScene.Request();req.scene.is_diff=True
        table=Pose();table.position.x=.15;table.position.z=-.025;table.orientation.w=1.
        rows=[('task2_table',table,[0.8,0.8,0.05])]
        for key in self.config['grids']:
            name='task3_object_'+key;p=self.pose_of(name)
            if p:rows.append((name,p,[0.04,0.04,0.04]))
        for name,p,size in rows:
            obj=CollisionObject();obj.header.frame_id='base';obj.id=name;obj.operation=CollisionObject.ADD
            primitive=SolidPrimitive();primitive.type=SolidPrimitive.BOX;primitive.dimensions=size
            obj.primitives=[primitive];obj.primitive_poses=[p];req.scene.world.collision_objects.append(obj)
        if not self.call(self.scene,req).success:raise RuntimeError('scene_initialization_failed')
        self.events.write('collision_scene_synchronized',objects=[r[0] for r in rows])

    def select(self,key):
        self.target_name='task3_object_'+key
        if not self.selector.wait_for_service(timeout_sec=8):raise RuntimeError('grasp_selector_unavailable')
        req=SetParameters.Request();param=Parameter();param.name='target_model';param.value=ParameterValue(type=4,string_value=self.target_name);req.parameters=[param]
        response=self.call(self.selector,req)
        if not response.results or not all(x.successful for x in response.results):raise RuntimeError('grasp_target_selection_rejected')
        self.apply_scene(world_operation=CollisionObject.REMOVE)

    def solve_ik(self,pose,seed):
        # Each path starts from a seed, but the first point may be in another fixed region.
        req=GetPositionIK.Request();ik=req.ik_request
        ik.group_name='arm';ik.ik_link_name='link6';ik.avoid_collisions=True
        ik.robot_state=self.robot_state(seed);ik.pose_stamped=pose;ik.timeout.sec=1
        response=self.call(self.ik,req,8)
        if response.error_code.val!=1:return None,f'IK error {response.error_code.val}',None
        values=dict(zip(response.solution.joint_state.name,response.solution.joint_state.position))
        solution=[values[name] for name in d.ARM_JOINTS]
        if any(not low<=value<=high for value,(low,high) in zip(solution,self.config['motion']['joint_limits'])):
            return None,'joint_limit',None
        if not 0<=solution[0]<=math.pi or abs(solution[3]-.086)>.6 or abs(solution[5]+1.187)>.6:
            return None,'unnatural_base_or_wrist_branch',None
        return solution,None,max(abs(a-b) for a,b in zip(solution,seed))

    @staticmethod
    def radial_reference(reference,xy):
        pose=copy.deepcopy(reference)
        yaw=math.atan2(xy[1],xy[0])-math.atan2(0.08,0.18)
        s,c=math.sin(yaw/2),math.cos(yaw/2)
        q=pose.pose.orientation;x,y,z,w=q.x,q.y,q.z,q.w
        q.x=c*x-s*y;q.y=c*y+s*x;q.z=c*z+s*w;q.w=c*w-s*z
        return pose

    def execute_target(self,target,label):
        # Home/approach moves use one monotone joint interpolation, never an
        # unconstrained sampling planner which can wrap behind the base.
        current=self.current_arm_positions()
        if current is None:return False,{'reason':'joint state unavailable'}
        count=max(1,math.ceil(max(abs(a-b) for a,b in zip(target,current))/0.035))
        chain=[[a+(b-a)*i/count for a,b in zip(current,target)] for i in range(1,count+1)]
        return self.execute_chain(chain,d.CLOSED if self.held else d.OPEN,label)

    def preflight(self,start_xy,end_xy):
        reference,reason=self.grasp_reference_pose()
        if reference is None:return None,{'ok':False,'reason':reason}
        safe=self.config['motion']['safe_object_z'];release=self.config['motion']['release_object_z']
        start=[(*start_xy,z) for z in [d.OBJECT_Z_M,safe]]
        end=[(*end_xy,z) for z in [safe,release]]
        start_points=list(reversed(d.interpolate_xyz(*start)))
        start_ref=self.radial_reference(reference,start_xy)
        seed=list(d.REFERENCE_GRASP_JOINTS)
        seed[0]+=math.atan2(start_xy[1],start_xy[0])-math.atan2(0.08,0.18)
        start_path,one=self.solve_object_path(start_points,seed,start_ref,'grid_vertical')
        if start_path is None:return None,{'ok':False,'chains':[one]}
        # Follow the short arc in the front half-plane at a constant height.
        # Keep each IK seed on the previous branch and reject discontinuities.
        a0,a1=math.atan2(start_xy[1],start_xy[0]),math.atan2(end_xy[1],end_xy[0])
        if not (0<a0<math.pi and 0<a1<math.pi):return None,{'ok':False,'reason':'target_outside_front_workspace'}
        r0,r1=math.hypot(*start_xy),math.hypot(*end_xy)
        count=max(2,math.ceil(abs(a1-a0)/0.025),math.ceil(abs(r1-r0)/0.003))
        transfer=[];cartesian=[];current=start_path[0];largest=0.
        for i in range(1,count+1):
            t=i/count;angle=a0+(a1-a0)*t;radius=r0+(r1-r0)*t
            xyz=(radius*math.cos(angle),radius*math.sin(angle),safe)
            pose=self.link6_pose_for_object_center(xyz,self.radial_reference(reference,xyz[:2]))
            solution,reason,delta=self.solve_ik(pose,current)
            if solution is None or delta>d.CHAIN_MAX_JOINT_STEP_RAD:
                return None,{'ok':False,'reason':reason or 'transfer_joint_branch_jump','index':i}
            largest=max(largest,delta);transfer.append(solution);cartesian.append(xyz);current=solution
        end_path,two=self.solve_object_path(d.interpolate_xyz(*end),current,self.radial_reference(reference,end_xy),'bin_vertical')
        if end_path is None:return None,{'ok':False,'chains':[one,two]}
        prepared={'start_vertical':start_path,'end_vertical':end_path,'transfer':transfer,
                  'start_grasp':start[0],'start_safe':start[1],'end_safe':end[0],'end_release':end[1]}
        self.prepared=prepared
        return prepared,{'ok':True,'chains':[one,two],'transfer_planner':'continuous radial IK, front arc, fixed height',
                         'transfer_xyz':cartesian,'transfer_joints':transfer,
                         'largest_transfer_joint_step_deg':math.degrees(largest),
                         'base_rotation_deg':math.degrees(a1-a0)}

    def reset_target(self,task_point):
        # Never teleport a perceived object into the gripper: check the physical scene.
        p=self.entity_position()
        return p is not None and d.distance(p,task_point)<=0.006,p

    def move_gripper(self,position,label):
        if self.fault=='grasp_failure' and position==d.CLOSED:
            self.events.write('fault_injection',fault='grasp_failure',method='keep fingers open; require physical grasp check to reject')
            return super().move_gripper(d.OPEN,'injected open gripper')
        return super().move_gripper(position,label)

    def trigger_service(self,client,label,require_success=True):
        result=super().trigger_service(client,label,require_success)
        if client==self.attach and result[0] is not None:self.held=True
        if client==self.detach and result[0] is not None:self.held=False
        return result

    def recover(self):
        if self.active_goal is not None:
            try:self.call_cancel()
            except Exception:pass
        if self.held:
            self.events.write('safe_stop',reason='object remains attached; movement canceled; no midair release')
            return False,'stopped_with_object_held'
        try:
            self.move_gripper(d.OPEN,'recovery_open')
            if self.prepared:
                ok,detail=self.execute_target(self.prepared['start_vertical'][0],'recovery_retreat')
                if not ok:return False,str(detail)
            self.sync_world()
            ok,detail=self.execute_target(d.HOME,'recovery_home')
            return ok,str(detail)
        except Exception as exc:return False,str(exc)

    def call_cancel(self):
        future=self.active_goal.cancel_goal_async()
        rclpy.spin_until_future_complete(self,future,timeout_sec=5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--request');p.add_argument('--output',required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--fault',default='')
    args=p.parse_args();rclpy.init(args=[],signal_handler_options=SignalHandlerOptions.NO);node=Motion(load_config(args.config),args.run_dir,args.fault)
    report={'ok':False,'status':'FAIL','safe_to_continue':True,'fault_injection':args.fault or None}
    code=1
    try:
        if not args.prepare_only:
            job=json.loads(Path(args.request).read_text());report.update(job)
        if not node.ready_full():raise RuntimeError('simulator_services_unavailable')
        node.sync_world()
        if args.prepare_only:
            report.update(ok=True,status='READY');code=0
        else:
            key=job['grid_id'];label=job['class_id'];slot=job['bin_slot']
            node.select(key)
            start=node.config['grids'][key]['xy'];end=node.config['bins'][label]['slots'][slot]
            if args.fault=='unreachable':
                node.events.write('fault_injection',fault='unreachable',method='request IK at [1,1] metres')
                start=[1.,1.]
            prepared,check=node.preflight(start,end);report['preflight']=check
            if prepared is None:raise RuntimeError('unreachable_or_collision_preflight')
            report['execution']=node.execute_pick_place(prepared,start,end)
            report.update(ok=True,status='SORTED',final_position=node.entity_position());code=0
    except (Exception,KeyboardInterrupt) as exc:
        report['error']=str(exc) or 'cancelled';report['steps_before_failure']=node.step_records
        if not args.prepare_only:
            safe,detail=node.recover();report.update(safe_to_continue=safe,recovery=detail)
        node.events.write('job_failed',error=report['error'],safe_to_continue=report['safe_to_continue'])
    finally:
        Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('RESULT '+json.dumps(report,ensure_ascii=False),flush=True)
        node.destroy_node();rclpy.try_shutdown()
    return code

if __name__=='__main__':sys.exit(main())
