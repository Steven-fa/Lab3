#!/usr/bin/env python3
"""Offline regression checks; these do not replace the live acceptance test."""
from pathlib import Path
import sys
import cv2
import numpy as np

bundle=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(bundle/'src/mecharm_task3/scripts'))
from common import load_config
from generate_world import bridge_model
from detector import detect_hsv

config=load_config(bundle/'src/mecharm_task3/config/sorting.yaml')
baseline=load_config(bundle.parent/'experiment3_sim_complete_20260914/src/mecharm_task3/config/sorting.yaml')
assert config['grids']==baseline['grids']
assert config['motion']==baseline['motion']
for label in ['red_cube','blue_cube']:
    assert config['bins'][label]==baseline['bins'][label]
    assert config['detector']['hsv_ranges'][label]==baseline['detector']['hsv_ranges'][label]
assert config['scene_labels']==['blue_cube','orange_bridge','red_cube','orange_bridge','red_cube','blue_cube']
model=bridge_model('test','0 0 .022','0.95 0.6 0.015 1')
collisions=model.findall('link/collision')
assert len(collisions)==3
volume=0
for collision in collisions:
    dims=list(map(float,collision.findtext('geometry/box/size').split()))
    pos=list(map(float,collision.findtext('pose').split()))[:3]
    assert all(abs(p)+s/2<=.020000001 for p,s in zip(pos,dims))
    # The underside centre must be actual free space.
    assert not all(abs(p-q)<s/2 for p,q,s in zip(pos,[0,0,-.01],dims))
    volume+=np.prod(dims)
assert abs(volume-(.04*.04*.012+2*.01*.04*.028))<1e-12
for hue,label in [(0,'red_cube'),(120,'blue_cube'),(19,'orange_bridge'),(60,'unknown')]:
    hsv=np.zeros((120,120,3),np.uint8)
    hsv[25:85,25:85]=[hue,240,230]
    boxes=detect_hsv(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR),config)
    assert len(boxes)==1 and boxes[0][0]==label and boxes[0][1]>=.65,boxes
assert config['task']['min_successes']==6
assert config['task']['required_bridge_successes']==2
print('PASS: unchanged grids/motion/red-blue rules; real hollow geometry; HSV classes; strict six-object/two-bridge criteria')
