#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D,Detection2DArray,ObjectHypothesisWithPose
from common import JsonLog,load_config,project_xy

def detect_hsv(frame,config):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); params=config['detector']
    known=np.zeros(frame.shape[:2],np.uint8); candidates=[]
    for label,ranges in params['hsv_ranges'].items():
        mask=np.zeros(frame.shape[:2],np.uint8)
        for low,high in ranges:mask|=cv2.inRange(hsv,np.array(low),np.array(high))
        known|=mask;candidates.append((label,mask))
    unknown=cv2.inRange(hsv,np.array([0,params['unknown_saturation'],70]),np.array([179,255,255]))
    unknown&=cv2.bitwise_not(known);candidates.append(('unknown',unknown))
    boxes=[]
    for label,mask in candidates:
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour)
            if area<params['min_area_px']:continue
            x,y,w,h=cv2.boundingRect(contour)
            if area>params.get('max_area_px',10000) or max(w/h,h/w)>params.get('max_aspect_ratio',1.8):continue
            # Axis-aligned box fill falsely gives a perfect 45-degree square
            # a score near 0.5. Measure fill in its rotated enclosing box.
            _,(rotated_w,rotated_h),_=cv2.minAreaRect(contour)
            score=min(0.99,area/max(rotated_w*rotated_h,1.0))
            boxes.append((label,score,x,y,w,h))
    return boxes

class Detector(Node):
    def __init__(self,args):
        super().__init__('task3_detector')
        self.config=load_config(args.config);self.bridge=CvBridge();self.frames=0
        self.log=JsonLog(args.run_dir,'detections.jsonl');self.last_log=0;self.state='WAIT_READY'
        self.backend=args.backend;self.model=None
        if self.backend=='yolo':
            if not args.weights:raise ValueError('YOLO backend requires --weights')
            from ultralytics import YOLO
            self.model=YOLO(args.weights)
        self.pub=self.create_publisher(Detection2DArray,'/task3/detections',10)
        self.images=self.create_publisher(Image,'/task3/annotated_image',qos_profile_sensor_data)
        self.create_subscription(Image,'/task3/camera/overhead/image_raw',self.on_image,qos_profile_sensor_data)
        self.create_subscription(String,'/task3/state',self.on_state,10)
        size=(self.config['camera']['width'],self.config['camera']['height']+80)
        self.video=cv2.VideoWriter(str(Path(args.run_dir)/'simulation_demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),self.config['camera']['rate'],size)
        if not self.video.isOpened():raise RuntimeError('Video writer could not open')
        self.folder=Path(args.run_dir);self.video_done=False;self.terminal_state=False
        self.log.write('detector_started',backend=self.backend,source='/task3/camera/overhead/image_raw',uses_scene_ground_truth=False)

    def on_state(self,msg):
        data=json.loads(msg.data);self.state=f"{data.get('state')} | sorted={data.get('successes',0)} | {data.get('detail','')}"
        if data.get('state') in ['DONE','STOPPED']:self.terminal_state=True

    def on_image(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        if self.model:
            results=self.model.predict(frame,conf=self.config['detector']['min_confidence'],verbose=False)[0]
            boxes=[]
            for box in results.boxes:
                x1,y1,x2,y2=box.xyxy[0].tolist();name=results.names[int(box.cls[0])]
                name={'red_cuboid':'red_cube','blue_cube':'blue_cube'}.get(name,'unknown')
                boxes.append((name,float(box.conf[0]),int(x1),int(y1),int(x2-x1),int(y2-y1)))
        else:boxes=detect_hsv(frame,self.config)
        message=Detection2DArray();message.header=msg.header
        annotated=frame.copy();records=[]
        for i,(label,score,x,y,w,h) in enumerate(boxes):
            det=Detection2D();det.header=msg.header;det.id=str(i)
            det.bbox.center.position.x=float(x+w/2);det.bbox.center.position.y=float(y+h/2)
            det.bbox.size_x=float(w);det.bbox.size_y=float(h)
            hypothesis=ObjectHypothesisWithPose();hypothesis.hypothesis.class_id=label;hypothesis.hypothesis.score=float(score)
            det.results=[hypothesis];message.detections.append(det)
            color=(0,230,230) if label=='unknown' else (0,230,0)
            cv2.rectangle(annotated,(x,y),(x+w,y+h),color,2)
            cv2.putText(annotated,f'{label} {score:.2f}',(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
            records.append({'class_id':label,'confidence':score,'bbox':[x,y,w,h]})
        for key,grid in self.config['grids'].items():
            u,v=project_xy(self.config,grid['xy'])
            cv2.putText(annotated,'G'+key,(int(u)-9,int(v)+43),cv2.FONT_HERSHEY_SIMPLEX,0.5,(240,240,240),1)
        self.pub.publish(message)
        out=self.bridge.cv2_to_imgmsg(annotated,'bgr8');out.header=msg.header;self.images.publish(out)
        canvas=cv2.copyMakeBorder(annotated,0,80,0,0,cv2.BORDER_CONSTANT,value=(22,22,22))
        h=annotated.shape[0]
        cv2.putText(canvas,'Experiment 3 | Actual Gazebo camera -> detection -> sorting',(15,h+25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(240,240,240),1)
        cv2.putText(canvas,self.state[:115],(15,h+55),cv2.FONT_HERSHEY_SIMPLEX,0.52,(120,240,180),1)
        if not self.video_done:
            self.video.write(canvas);self.frames+=1
            if self.terminal_state:
                cv2.imwrite(str(self.folder/'camera_final.png'),canvas)
                self.video.release();self.video_done=True
                (self.folder/'recording.json').write_text(json.dumps({'frames':self.frames,'fps':self.config['camera']['rate'],'source':'actual Gazebo sensor images; detection boxes and task state overlaid'}),encoding='utf-8')
        if self.frames==8:cv2.imwrite(str(self.folder/'camera_initial.png'),canvas)
        if self.frames%16==0:cv2.imwrite(str(self.folder/'camera_latest.png'),canvas)
        if time.monotonic()-self.last_log>=0.5:
            self.log.write('image_detection',stamp={'sec':msg.header.stamp.sec,'nanosec':msg.header.stamp.nanosec},detections=records,frame=self.frames)
            self.last_log=time.monotonic()

    def close(self):
        self.video.release();self.log.write('recording_closed',frames=self.frames)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--backend',default='hsv',choices=['hsv','yolo']);parser.add_argument('--weights',default='')
    args,rosargs=parser.parse_known_args();rclpy.init(args=rosargs);node=Detector(args)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.close();node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
