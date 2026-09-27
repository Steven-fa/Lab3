#!/usr/bin/env python3
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as E
from ament_index_python.packages import get_package_share_directory
from common import load_config

def box_model(name,xyz,size,color,static=True,collision=True):
    physical=f'<collision name="collision"><geometry><box><size>{size}</size></box></geometry><surface><friction><ode><mu>1</mu><mu2>1</mu2></ode></friction></surface></collision>' if collision else ''
    return E.fromstring(f'''<model name="{name}"><static>{str(static).lower()}</static><pose>{xyz} 0 0 0</pose><link name="link"><inertial><mass>0.05</mass><inertia><ixx>0.00001333</ixx><iyy>0.00001333</iyy><izz>0.00001333</izz></inertia></inertial>{physical}<visual name="visual"><geometry><box><size>{size}</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual></link></model>''')

def generate(config_path,scenario,seed,run_dir):
    config=load_config(config_path); out=Path(run_dir);out.mkdir(parents=True,exist_ok=True)
    original=Path(get_package_share_directory('mecharm_task2'))/'worlds/task2.world'
    tree=E.parse(original); world=tree.getroot().find('world')
    for model in list(world.findall('model')):
        if model.get('name')!='task2_table':world.remove(model)
    world.set('name','task3_sorting')
    world.find('gui/camera/pose').text='0.62 -0.62 0.57 0 0.58 2.35'
    labels=['red_cube']*3+['blue_cube']*3
    random.Random(seed).shuffle(labels)
    faults={}
    if scenario=='exceptions':
        labels=['red_cube','unknown',None,'blue_cube','red_cube','blue_cube']
        faults={'5':'unreachable','6':'grasp_failure'}
    truth=[]
    colors={'red_cube':'0.9 0.015 0.015 1','blue_cube':'0.015 0.06 0.95 1','unknown':'0.03 0.85 0.08 1'}
    for i,(key,grid) in enumerate(config['grids'].items()):
        x,y=grid['xy']; width=2*grid['half_width']
        for axis in range(2):
            for side in [-1,1]:
                px=x+(side*width/2 if axis==0 else 0);py=y+(side*width/2 if axis==1 else 0)
                size=f'0.0015 {width} 0.0005' if axis==0 else f'{width} 0.0015 0.0005'
                world.append(box_model(f'grid_{key}_{axis}_{side}',f'{px} {py} 0.0003',size,'0.85 0.85 0.85 1',collision=False))
        label=labels[i];name=f'task3_object_{key}'
        if label:
            cube=box_model(name,f'{x} {y} 0.022','0.04 0.04 0.04',colors[label],False)
            # Fixed workcell fixtures align the cube faces with the same grasp
            # orientation as the reference. A diagonal 40 mm cube is wider
            # than this gripper's opening; arbitrary yaw is outside this task.
            yaw=math.atan2(y,x)-math.atan2(0.08,0.18)
            cube.find('pose').text=f'{x} {y} 0.022 0 0 {yaw}'
            world.append(cube)
        truth.append({'grid':key,'name':name,'class_id':label,'initial_xy':[x,y],
                      'fixture_yaw_rad':math.atan2(y,x)-math.atan2(0.08,0.18)})
    for label,bin in config['bins'].items():
        xs,ys=zip(*bin['slots']);x=(min(xs)+max(xs))/2;y=(min(ys)+max(ys))/2
        color='0.45 0.33 0.33 1' if label=='red_cube' else '0.33 0.37 0.45 1'
        size=f'{max(xs)-min(xs)+0.056} {max(ys)-min(ys)+0.056} 0.0001'
        world.append(box_model(bin['id'],f'{x} {y} 0.0001',size,color,collision=False))
    c=config['camera'];xyz=' '.join(map(str,c['position']))
    camera=E.fromstring(f'''<model name="task3_overhead"><static>true</static><pose>{xyz} 0 1.57079632679 1.57079632679</pose><link name="camera_link"><sensor name="overhead" type="camera"><always_on>true</always_on><update_rate>{c['rate']}</update_rate><visualize>false</visualize><camera><horizontal_fov>{c['hfov']}</horizontal_fov><image><width>{c['width']}</width><height>{c['height']}</height><format>R8G8B8</format></image><clip><near>0.02</near><far>5</far></clip></camera><plugin name="task3_camera" filename="libgazebo_ros_camera.so"><ros><namespace>/task3/camera</namespace><remapping>image_raw:=image_raw</remapping><remapping>camera_info:=camera_info</remapping></ros><camera_name>overhead</camera_name><frame_name>task3_camera_optical</frame_name></plugin></sensor></link></model>''')
    world.append(camera)
    path=out/'task3.world';tree.write(path,encoding='utf-8',xml_declaration=True)
    (out/'ground_truth.json').write_text(json.dumps({'scenario':scenario,'seed':seed,'objects':truth,'faults':faults},indent=2),encoding='utf-8')
    return str(path)
