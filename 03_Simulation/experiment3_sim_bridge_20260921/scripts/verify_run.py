#!/usr/bin/env python3
"""Cross-check saved independent evaluation, real SDF geometry, and playable videos."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2

folder=Path(sys.argv[1])
acceptance=json.loads((folder/'acceptance.json').read_text())
task=json.loads((folder/'task_result.json').read_text())
world=ET.parse(folder/'task3.world').getroot().find('world')
geometry={}
for item in acceptance['details']:
    model=world.find("model[@name='"+item['name']+"']")
    geometry[item['grid']]={'shape':item['shape'],'collisions':len(model.findall('link/collision'))}
    assert geometry[item['grid']]['collisions']==(3 if item['shape']=='bridge' else 1)
videos={}
for name in ['simulation_demo.mp4','side_view.mp4']:
    cap=cv2.VideoCapture(str(folder/name))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    decoded=[]
    for index in [0,count//2,count-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
        decoded.append(bool(ok and frame is not None))
    cap.release()
    videos[name]={'frames':count,'fps':fps,'start_middle_end_decoded':decoded}
    assert count>0 and fps>0 and all(decoded),videos[name]
assert acceptance['status']=='PASS'
assert acceptance['correct_count']==6 and acceptance['bridge_correct_count']==2
assert acceptance['safe'] and acceptance['task_state']=='DONE'
assert len(task['jobs'])==6 and all(job['ok'] for job in task['jobs'])
assert all(row['correct_bin'] for row in acceptance['details'])
print(json.dumps({'verified':True,'geometry':geometry,'videos':videos,
                  'note':'encoded video duration is not wall-clock duration'},indent=2))
