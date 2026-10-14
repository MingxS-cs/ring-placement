import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
out=Path('outputs/v4_holdout_review');a=json.loads((out/'sequential_index_audit_before_correction.json').read_text());b=json.loads((out/'ffprobe_sequential_pts_audit.json').read_text());p=Path('data/v4_holdout_annotations.json');h=hashlib.sha256(p.read_bytes()).hexdigest();receipt=json.loads((out/'freeze_receipt.json').read_text());assert h==receipt['annotation_sha256']
for s in a['samples']:
 assert s['matching_sequential_ordinals']==[s['requested_frame']]
 assert s['requested_pts_s']==s['sequential_pts_s_at_requested_ordinal']
assert b['status']=='passed' and b['checked_samples']==35 and b['max_absolute_pts_difference_s']<5e-7
summary={'schema_version':1,'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Counts, source IDs and integrity findings only; detailed frame-index/PTS audits remain withheld with the annotations until selected-model freeze.',
 'annotation_sha256':h,'source_sha256':receipt['source_sha256'],'source_total_sequential_frame_count':a['source_total_sequential_frames'],
 'valid_source_ids':['E20','E21','E22'],'checked_raw_evidence_images':35,'same_ordinal_unique_image_hash_matches':35,
 'ordinal_index_mismatches':0,'requested_frame_n_matching_ordinal_n_minus_1':0,
 'sequential_opencv_pts_matches':35,'ffprobe_display_order_best_effort_pts_matches':35,
 'ffprobe_pts_check_tolerance_s':5e-7,'max_ffprobe_pts_rounding_difference_s':b['max_absolute_pts_difference_s'],
 'annotation_correction_applied':False,'annotation_and_image_hashes_unchanged':True,
 'conclusion':'Every locked evidence image uniquely matches the same zero-based sequential decoded ordinal. Recorded PTS agrees with both sequential OpenCV decoding and ffprobe display-order presentation timestamps. The later PTS offset relative to ordinal/fps exists in source timestamps and is not a seek index mismatch.',
 'episodes':[{'source_id':e['source_id'],'samples_checked':e['measured_samples'],'ordinal_mismatches':0,'pts_mismatches':0} for e in receipt['episodes']],
 'detailed_audit_hashes':{'sequential_index_audit':hashlib.sha256((out/'sequential_index_audit_before_correction.json').read_bytes()).hexdigest(),'ffprobe_pts_audit':hashlib.sha256((out/'ffprobe_sequential_pts_audit.json').read_bytes()).hexdigest()}}
(out/'sequential_index_receipt.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
receipt['sequential_index_verification']={'status':'passed','receipt_path':'outputs/v4_holdout_review/sequential_index_receipt.json','receipt_sha256':hashlib.sha256((out/'sequential_index_receipt.json').read_bytes()).hexdigest(),'checked_raw_images':35,'ordinal_mismatches':0,'pts_mismatches':0,'annotation_correction_applied':False,'verified_utc':summary['verified_utc']}
(out/'freeze_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
