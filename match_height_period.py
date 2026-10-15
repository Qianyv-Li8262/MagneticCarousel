import json,numpy as np
import refine_height_response as r
if __name__=='__main__':
 cases=[('long_period',2.48587994094,26.91),('short_period_at_same_disk_rpm',2.41625700075,26.68)];records=[]
 for label,target,h in cases:
  r.make_table(h);run=r.a.run(h,600,.00025,True);record=dict(label=label,target_period_s=target,height_mm=h,run=run,field_stats=r.a.fieldstats(h),validation=r.validate(h));records.append(record);(r.a.OUT/'matched_periods.json').write_text(json.dumps(records,indent=2))
 print(json.dumps([{k:q[k] for k in ['label','target_period_s','height_mm']} for q in records]),flush=True)
