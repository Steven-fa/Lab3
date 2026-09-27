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

def bridge_model(name, xyz, color):
    # 40 mm outer envelope, 20 mm wide x 28 mm high through-opening.
    # A single rigid body with three disjoint boxes, not a solid recoloured cube.
    model=box_model(name,xyz,'0.04 0.04 0.04',color,False,False)
    link=model.find('link');link.remove(link.find('visual'))
    for part,size,pose in [('beam','0.04 0.04 0.012','0 0 0.014'),
                           ('left_leg','0.01 0.04 0.028','-0.015 0 -0.006'),
                           ('right_leg','0.01 0.04 0.028','0.015 0 -0.006')]:
        source=box_model(part,'0 0 0',size,color,False).find('link')
        for tag in ['collision','visual']:
            element=source.find(tag);element.set('name',part+'_'+tag)
            E.SubElement(element,'pose').text=pose+' 0 0 0';link.append(element)
    # Uniform-density composite mass/inertia including its elevated centre of mass.
    parts=[(.04,.04,.012,0.,0.,.014),(.01,.04,.028,-.015,0.,-.006),(.01,.04,.028,.015,0.,-.006)]
    volume=sum(a*b*c for a,b,c,_,_,_ in parts)
    masses=[.05*a*b*c/volume for a,b,c,_,_,_ in parts]
    cz=sum(m*p[5] for m,p in zip(masses,parts))/.05
    inertial=link.find('inertial');E.SubElement(inertial,'pose').text=f'0 0 {cz} 0 0 0'
    moments=[0.,0.,0.]
    for m,(a,b,c,x,y,z) in zip(masses,parts):
        moments[0]+=m*((b*b+c*c)/12+y*y+(z-cz)**2)
        moments[1]+=m*((a*a+c*c)/12+x*x+(z-cz)**2)
        moments[2]+=m*((a*a+b*b)/12+x*x+y*y)
    for tag,value in zip(['ixx','iyy','izz'],moments):inertial.find('inertia/'+tag).text=str(value)
    return model

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
    labels=config.get('scene_labels',labels)
    faults={}
    if scenario=='exceptions':
        labels=['red_cube','unknown',None,'blue_cube','red_cube','blue_cube']
        faults={'5':'unreachable','6':'grasp_failure'}
    truth=[]
    colors={'red_cube':'0.9 0.015 0.015 1','blue_cube':'0.015 0.06 0.95 1','orange_bridge':'0.95 0.6 0.015 1','unknown':'0.03 0.85 0.08 1'}
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
            if label=='orange_bridge':cube=bridge_model(name,f'{x} {y} 0.022',colors[label])
            # Fixed workcell fixtures align the cube faces with the same grasp
            # orientation as the reference. A diagonal 40 mm cube is wider
            # than this gripper's opening; arbitrary yaw is outside this task.
            yaw=math.atan2(y,x)-math.atan2(0.08,0.18)
            cube.find('pose').text=f'{x} {y} 0.022 0 0 {yaw}'
            world.append(cube)
        truth.append({'grid':key,'name':name,'class_id':label,'initial_xy':[x,y],
                      'shape':'bridge' if label=='orange_bridge' else 'cube',
                      'fixture_yaw_rad':math.atan2(y,x)-math.atan2(0.08,0.18)})
    for label,bin in config['bins'].items():
        xs,ys=zip(*bin['slots']);x=(min(xs)+max(xs))/2;y=(min(ys)+max(ys))/2
        color='0.45 0.33 0.33 1' if label=='red_cube' else '0.33 0.37 0.45 1'
        if label=='orange_bridge':color='0.45 0.40 0.33 1'
        size=f'{max(xs)-min(xs)+0.056} {max(ys)-min(ys)+0.056} 0.0001'
        world.append(box_model(bin['id'],f'{x} {y} 0.0001',size,color,collision=False))
    c=config['camera'];xyz=' '.join(map(str,c['position']))
    camera=E.fromstring(f'''<model name="task3_overhead"><static>true</static><pose>{xyz} 0 1.57079632679 1.57079632679</pose><link name="camera_link"><sensor name="overhead" type="camera"><always_on>true</always_on><update_rate>{c['rate']}</update_rate><visualize>false</visualize><camera><horizontal_fov>{c['hfov']}</horizontal_fov><image><width>{c['width']}</width><height>{c['height']}</height><format>R8G8B8</format></image><clip><near>0.02</near><far>5</far></clip></camera><plugin name="task3_camera" filename="libgazebo_ros_camera.so"><ros><namespace>/task3/camera</namespace><remapping>image_raw:=image_raw</remapping><remapping>camera_info:=camera_info</remapping></ros><camera_name>overhead</camera_name><frame_name>task3_camera_optical</frame_name></plugin></sensor></link></model>''')
    world.append(camera)
    path=out/'task3.world';tree.write(path,encoding='utf-8',xml_declaration=True)
    (out/'ground_truth.json').write_text(json.dumps({'scenario':scenario,'seed':seed,'objects':truth,'faults':faults},indent=2),encoding='utf-8')
    return str(path)
