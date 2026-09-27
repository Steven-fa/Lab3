#!/usr/bin/env python3
"""Read-only observation camera: no detections, scene edits to objects, or motion commands."""
import json
from pathlib import Path
import sys
import time
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from gazebo_msgs.srv import SpawnEntity
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rclpy.init()
node=Node('bridge_side_view_recorder')
folder=Path(sys.argv[1]);bridge=CvBridge();frames=0;writer=None
client=node.create_client(SpawnEntity,'/spawn_entity')
if not client.wait_for_service(timeout_sec=90):raise RuntimeError('spawn unavailable')
request=SpawnEntity.Request();request.name='bridge_observation_camera'
request.xml='''<sdf version="1.6"><model name="bridge_observation_camera"><static>true</static>
<pose>0.30 0.40 0.24 0 0.49 -2.391</pose><link name="camera_link">
<sensor name="side" type="camera"><always_on>true</always_on><update_rate>5</update_rate>
<camera><horizontal_fov>0.9</horizontal_fov><image><width>1280</width><height>960</height><format>R8G8B8</format></image>
<clip><near>0.02</near><far>5</far></clip></camera>
<plugin name="bridge_side_camera" filename="libgazebo_ros_camera.so"><ros><namespace>/bridge_observer</namespace></ros>
<camera_name>side</camera_name><frame_name>bridge_side_optical</frame_name></plugin></sensor></link></model></sdf>'''
future=client.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=20)
if not future.done() or not future.result().success:raise RuntimeError('observer spawn failed')
print('OBSERVER_STARTED',flush=True)

def on_image(msg):
    global frames,writer
    frame=bridge.imgmsg_to_cv2(msg,'bgr8')
    if writer is None:
        writer=cv2.VideoWriter(str(folder/'side_view.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(frame.shape[1],frame.shape[0]))
        if not writer.isOpened():raise RuntimeError('video writer unavailable')
    writer.write(frame);frames+=1
    if frames==1:cv2.imwrite(str(folder/'side_initial.png'),frame)
    if frames%10==0:cv2.imwrite(str(folder/'side_latest.png'),frame)
    if (folder/'acceptance.json').exists():cv2.imwrite(str(folder/'side_final.png'),frame)

subscription=node.create_subscription(Image,'/bridge_observer/side/image_raw',on_image,qos_profile_sensor_data)
start=time.monotonic()
try:
    while time.monotonic()-start<1200:
        rclpy.spin_once(node,timeout_sec=.2)
        if (folder/'side_final.png').exists():break
except KeyboardInterrupt:pass
finally:
    if writer:writer.release()
    (folder/'side_recording.json').write_text(json.dumps({'frames':frames,'encoded_fps':5,'role':'observation only; started after primary simulation launch'}))
    node.destroy_node();rclpy.try_shutdown()
    print('OBSERVER_FRAMES '+str(frames),flush=True)
