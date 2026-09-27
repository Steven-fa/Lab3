"""Summarize executed transfer samples; never replace simulation acceptance."""
import json
import math
from pathlib import Path
import sys

folder=Path(sys.argv[1])
events=[json.loads(line) for line in (folder/'motion_events.jsonl').read_text().splitlines()]
samples=[json.loads(line) for line in (folder/'safety.jsonl').read_text().splitlines()]
samples=[x for x in samples if x['event']=='joint_sample']
rows=[]
for path in sorted((folder/'jobs').glob('*.json')):
 job=json.loads(path.read_text())
 if not job.get('execution',{}).get('ok'):continue
 target='task3_object_'+job['grid_id']
 start=next(x['monotonic'] for x in events if x.get('target')==target and x.get('label')=='cube_lifted')
 end=next(x['monotonic'] for x in events if x.get('target')==target and x.get('label')=='transfer_to_end')
 actual=[x['positions'] for x in samples if start<=x['monotonic']<=end]
 plan=job['preflight'];angles=list(zip(*plan['transfer_joints']))
 joints=list(actual[0]) if actual else []
 actual_ranges={k:[round(math.degrees(min(s[k] for s in actual)),3),round(math.degrees(max(s[k] for s in actual)),3)] for k in joints}
 row={'grid_id':job['grid_id'],'class_id':job['class_id'],'bin_slot':job['bin_slot'],
      'actual_samples_during_transfer':len(actual),'actual_joint_ranges_deg':actual_ranges,
      'planned_base_rotation_deg':round(plan['base_rotation_deg'],3),
      'planned_max_adjacent_step_deg':round(plan['largest_transfer_joint_step_deg'],3),
      'planned_front_y_min_m':round(min(p[1] for p in plan['transfer_xyz']),6),
      'planned_z_range_m':[min(p[2] for p in plan['transfer_xyz']),max(p[2] for p in plan['transfer_xyz'])],
      'placed_xy_error_mm':job['execution']['final_xy_error_mm']}
 rows.append(row)
result={'note':'Plan coordinates and actual joint telemetry are distinguished; full safety and correctness are in acceptance.json.','jobs':rows}
(folder/'trajectory_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
