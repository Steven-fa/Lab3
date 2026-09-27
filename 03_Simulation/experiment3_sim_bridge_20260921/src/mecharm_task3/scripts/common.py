#!/usr/bin/env python3
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import yaml

def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--scenario',default='normal')
    p.add_argument('--seed',type=int,default=42)
    return p.parse_known_args()

def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))

class JsonLog:
    def __init__(self,folder,name):
        Path(folder).mkdir(parents=True,exist_ok=True)
        self.file=(Path(folder)/name).open('a',encoding='utf-8',buffering=1)
    def write(self,event,**data):
        row={'time':datetime.now().astimezone().isoformat(),'monotonic':time.monotonic(),'event':event,**data}
        self.file.write(json.dumps(row,ensure_ascii=False)+'\n')
        return row

def project_xy(config,xy,z=0.04):
    c=config['camera']; cx,cy,cz=c['position']
    focal=c['width']/(2*math.tan(c['hfov']/2))
    return [c['width']/2+focal*(xy[0]-cx)/(cz-z),c['height']/2-focal*(xy[1]-cy)/(cz-z)]

def grid_for_pixel(config,u,v):
    c=config['camera']; focal=c['width']/(2*math.tan(c['hfov']/2))
    candidates=[]
    for key,grid in config['grids'].items():
        center=project_xy(config,grid['xy'])
        half=focal*grid['half_width']/(c['position'][2]-0.04)
        if abs(u-center[0])<=half and abs(v-center[1])<=half:candidates.append(key)
    return candidates[0] if len(candidates)==1 else None

def snapshot_grid(config,detections):
    result={key:[] for key in config['grids']}
    for det in detections.detections:
        key=grid_for_pixel(config,det.bbox.center.position.x,det.bbox.center.position.y)
        if key is not None and det.results:
            h=max(det.results,key=lambda h:h.hypothesis.score).hypothesis
            result[key].append({'class_id':h.class_id,'score':float(h.score),
                                'bbox':[det.bbox.center.position.x,det.bbox.center.position.y,det.bbox.size_x,det.bbox.size_y]})
    return result
