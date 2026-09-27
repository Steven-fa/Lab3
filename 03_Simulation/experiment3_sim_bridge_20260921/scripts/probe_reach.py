#!/usr/bin/env python3
"""Read-only IK/state-validity diagnostics: never publish motion or scene updates."""
import json
import math
from pathlib import Path
import sys
import rclpy
from moveit_msgs.srv import GetPositionIK
bundle=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(bundle/'src/mecharm_task3/scripts'))
import dynamic_affine_pick_place as d
from task3_motion import Motion
from common import load_config
rclpy.init()
node=Motion(load_config(bundle/'src/mecharm_task3/config/sorting.yaml'),sys.argv[1])
if not node.fk.wait_for_service(timeout_sec=20):raise RuntimeError('FK unavailable')
reference,reason=node.grasp_reference_pose()
if reference is None:raise RuntimeError(reason)
points=[(-.094,.164)] if '--candidate' in sys.argv else [(-.075,.205),(-.1,.16),(-.1,.17),(-.12,.16),(-.065,.19),(-.09,.17)]
for xy in points:
    for z in [.065,.055,.045,.04]:
        req=GetPositionIK.Request();ik=req.ik_request
        ik.group_name='arm';ik.ik_link_name='link6';ik.avoid_collisions=False
        seed=list(d.REFERENCE_GRASP_JOINTS);seed[0]+=math.atan2(xy[1],xy[0])-math.atan2(.08,.18)
        ik.robot_state=node.robot_state(seed)
        ik.pose_stamped=node.link6_pose_for_object_center((*xy,z),node.radial_reference(reference,xy))
        ik.timeout.sec=1
        result=node.call(node.ik,req,5)
        row={'xy':xy,'z':z,'ik':result.error_code.val}
        if result.error_code.val==1:
            values=dict(zip(result.solution.joint_state.name,result.solution.joint_state.position))
            joints=[values[k] for k in d.ARM_JOINTS]
            row['joints']=joints
            row['valid'],row['contacts']=node.state_is_valid(joints,d.OPEN)
        print(json.dumps(row),flush=True)
node.destroy_node();rclpy.try_shutdown()
