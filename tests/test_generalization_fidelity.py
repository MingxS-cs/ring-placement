"""Synthetic tests only: this suite never opens the real blind annotations."""
import importlib
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ring_learning.dataset import split_episodes
from ring_learning.generalization_policy import DirectionRidgePolicy, save_generalization_policy, source_signature
from ring_learning.motion import build_path
from ring_learning.video_policy import ShapeNet, save_video_policy


def module():
    assert importlib.util.find_spec('ring_learning.generalization_fidelity') is not None, 'Fidelity evaluator is not implemented'
    return importlib.import_module('ring_learning.generalization_fidelity')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, sort_keys=True, allow_nan=False))


def item(source, progress, errors):
    p=np.array(progress,float)
    xy=np.c_[p,np.array(errors,float)]
    weights=np.r_[(p[1]-p[0])/2,(p[2:]-p[:-2])/2,(p[-1]-p[-2])/2]
    return dict(source=source,progress=p,xy=xy,start=xy[0],goal=xy[-1],weights=weights,
        source_frames=np.arange(len(p)),reference_angle=0.,sample_kind=['observed']*len(p))


class Straight:
    def __init__(self):self.seen=[]
    def waypoint(self,start,goal,progress,**kwargs):
        self.seen.append(float(progress))
        return (1-progress)*np.asarray(start)+progress*np.asarray(goal)


@pytest.fixture
def frozen(tmp_path):
    source=tmp_path/'source';source.mkdir();video=source/'synthetic.mp4';video.write_bytes(b'synthetic source bytes, not a real video')
    source_sha=sha(video)
    episodes=[dict(episode_id=f'E{i:02d}',scene_group=3 if i>=20 else (1 if i<=11 else 2),
        source_frame_start=i*100,source_frame_terminal=i*100+90) for i in range(1,25)]
    real=tmp_path/'endpoints.json';write_json(real,dict(source=dict(basename=video.name,sha256=source_sha),episodes=episodes))
    splits={name:sorted(e['episode_id'] for e in group) for name,group in split_episodes(episodes).items()}
    devel=tmp_path/'development.json';write_json(devel,dict(source_sha256=source_sha,episodes=[]))
    protocol=tmp_path/'protocol.md';protocol.write_text('Frozen source-LOPO selection. Validation and holdout do not select weights.')
    checkpoint=tmp_path/'policy.json'
    metadata=dict(method='real_video_direction_conditioned_ridge',seed=20261014,training_sources=splits['train'],
        source_splits=splits,development_annotations_sha256=sha(devel),real_episodes_sha256=sha(real),
        original_video_sha256=source_sha,implementation_sources=source_signature(),selection=dict(
            selected_degree=0,selected_alpha=.2,validation_used_for_selection=False,method='training_source_leave_one_out'))
    save_generalization_policy(checkpoint,DirectionRidgePolicy(np.zeros((3,2)),np.zeros(2),np.ones(2)),metadata)
    v3=tmp_path/'v3.pt';legacy=ShapeNet(hidden=2)
    with torch.no_grad():
        for p in legacy.parameters():p.zero_()
    save_video_policy(v3,legacy,dict(original_video_sha256=source_sha,training_sources=['E01','E02']))
    records=[]
    for source_id,n in zip(('E20','E21','E22'),(12,12,11)):
        original=next(e for e in episodes if e['episode_id']==source_id)
        q=np.linspace(0,1,n);xy=np.c_[-.10*(1-q),.012*np.sin(np.pi*q)]
        frames=list(range(original['source_frame_start']+10,original['source_frame_start']+10+n))
        centers=xy*np.array([1,-1])*(200/.05)+[500,400]
        samples=[dict(source_frame=f,source_pts_s=f/30,center_px=c.tolist(),kind='manual_visual') for f,c in zip(frames,centers)]
        records.append(dict(episode_id=source_id,status='accepted',source_frame_start=original['source_frame_start'],
            source_frame_terminal=original['source_frame_terminal'],bearing_center_px=[500,400],bearing_outer_diameter_px=200,
            samples=samples,path=build_path(xy,frames)))
    holdout=tmp_path/'synthetic_holdout.json';write_json(holdout,dict(schema_version=1,source_sha256=source_sha,episodes=records))
    receipt=tmp_path/'freeze.json'
    value=dict(schema_version=1,selected_checkpoint_sha256=sha(checkpoint),training_protocol_path=str(protocol),
        training_protocol_sha256=sha(protocol),development_annotations_path=str(devel),development_annotations_sha256=sha(devel),
        real_episodes_path=str(real),real_episodes_sha256=sha(real),original_video_sha256=source_sha,
        holdout_annotations_sha256=sha(holdout),training_sources=splits['train'],holdout_sources=['E20','E21','E22'],
        v3_checkpoint_path=str(v3),v3_checkpoint_sha256=sha(v3),holdout_observed_points=35,holdout_intermediate_points=29)
    write_json(receipt,value)
    return dict(checkpoint=checkpoint,receipt=receipt,holdout=holdout,v3=v3,freeze=value,metadata=metadata,
        output=tmp_path/'result',records=records,protocol=protocol,devel=devel,real=real)


def test_arc_weights_and_interior_renormalization_are_source_uniform():
    m=module();a=item('A',[0,.1,1],[0,.010,0]);b=item('B',[0,.8,.9,1],[0,.030,.030,0])
    result=m.fidelity_metrics(Straight(),Straight(),[a,b])
    straight=result['methods']['straight']
    assert straight['source_uniform_mean_xy_error_m']==pytest.approx((.005+.0165)/2)
    assert straight['source_uniform_mean_intermediate_xy_error_m']==pytest.approx(.020)
    assert result['observed_points']==7 and result['intermediate_points']==3
    assert result['per_source'][0]['methods']['straight']['intermediate_arc_weights']==pytest.approx([1.])


def test_all_three_predictions_share_the_observed_arc_progress():
    m=module();v4=Straight();v3=Straight();a=item('A',[0,.07,.62,1],[0,.02,.01,0])
    result=m.fidelity_metrics(v4,v3,[a]);row=result['per_source'][0]
    assert v4.seen==v3.seen==a['progress'].tolist()
    assert row['observed_relative_xy_m']==a['xy'].tolist()
    assert row['observed_arc_progress']==a['progress'].tolist()
    assert set(row['predicted_relative_xy_m'])=={'v4','straight','v3'}
    assert 'not human action timing' in result['progress_interpretation']


def test_coordinate_perturbation_preserves_true_anchors_and_recomputes_arc_progress():
    m=module();a=item('A',[0,.1,.8,1],[0,.015,.005,0]);record={'bearing_outer_diameter_px':200}
    first=m.perturb_observed_path(a,record,np.random.default_rng(20261021),5)
    second=m.perturb_observed_path(a,record,np.random.default_rng(20261021),5)
    assert np.array_equal(first['xy'],second['xy'])
    assert np.array_equal(first['xy'][[0,-1]],a['xy'][[0,-1]])
    assert np.max(np.abs(first['xy'][1:-1]-a['xy'][1:-1]))<=5*.05/200
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(first['xy'],axis=0),axis=1))]
    assert first['progress']==pytest.approx(arc/arc[-1])
    assert not np.allclose(first['progress'],a['progress'])
    assert a['xy'][1,1]==.015


@pytest.mark.parametrize('bad_key', ['selected_checkpoint_sha256','training_protocol_sha256',
    'development_annotations_sha256','holdout_annotations_sha256','v3_checkpoint_sha256'])
def test_freeze_mismatch_rejects_before_holdout_labels_are_parsed(frozen,monkeypatch,bad_key):
    m=module();frozen['freeze'][bad_key]='0'*64;write_json(frozen['receipt'],frozen['freeze'])
    original=Path.read_text
    def guarded(path,*args,**kwargs):
        assert path.resolve()!=frozen['holdout'].resolve(), 'Holdout labels were opened before freeze validation'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    with pytest.raises(ValueError,match='freeze|frozen|hash|provenance'):
        m.evaluate_fidelity(frozen['checkpoint'],frozen['receipt'],frozen['holdout'],frozen['output'],v3_checkpoint=frozen['v3'])
    assert not frozen['output'].exists()


def test_valid_frozen_synthetic_evaluation_reports_35_points_and_reproducible_measurement_sensitivity(frozen):
    m=module();result=m.evaluate_fidelity(frozen['checkpoint'],frozen['receipt'],frozen['holdout'],frozen['output'],v3_checkpoint=frozen['v3'])
    assert result['observed_points']==35 and result['intermediate_points']==29
    assert result['source_episode_ids']==['E20','E21','E22']
    assert all(len(row['observed_relative_xy_m'])==len(row['observed_arc_progress']) for row in result['per_source'])
    assert result['methods']['v4']==result['methods']['straight']==result['methods']['v3']
    sensitivity=result['measurement_sensitivity']
    assert sensitivity['draws_per_radius']==100 and sensitivity['seed']==20261021
    assert sensitivity['square_uniform_radius_px']==[5,8]
    assert sensitivity['endpoint_anchors_fixed'] and sensitivity['arc_progress_recomputed']
    assert 'not a confidence interval' in sensitivity['interpretation']
    assert len(sensitivity['radius_results'][0]['draws'])==100
    second=m.evaluate_fidelity(frozen['checkpoint'],frozen['receipt'],frozen['holdout'],frozen['output'].parent/'second',v3_checkpoint=frozen['v3'])
    assert second['measurement_sensitivity']==sensitivity
    assert (frozen['output']/'fidelity.json').is_file()


@pytest.mark.parametrize('mutation,expected', [('outside','temporal bounds'),('derived','lineage'),('interpolated','observed')])
def test_holdout_observation_lineage_is_checked_after_freeze(frozen,mutation,expected):
    m=module();payload=json.loads(frozen['holdout'].read_text());r=payload['episodes'][0]
    if mutation=='outside':r['samples'][0]['source_frame']=r['source_frame_start']-1;r['path']['sample_source_frames'][0]=r['source_frame_start']-1
    elif mutation=='derived':r['path']['relative_xy_m'][1][1]+=.001
    else:r['samples'][1]['kind']='short_gap_interpolated'
    write_json(frozen['holdout'],payload);frozen['freeze']['holdout_annotations_sha256']=sha(frozen['holdout']);write_json(frozen['receipt'],frozen['freeze'])
    with pytest.raises(ValueError,match=expected):
        m.evaluate_fidelity(frozen['checkpoint'],frozen['receipt'],frozen['holdout'],frozen['output'],v3_checkpoint=frozen['v3'])


def test_training_sources_cannot_overlap_reserved_holdout_even_with_updated_checkpoint_hash(frozen):
    m=module();value=json.loads(frozen['checkpoint'].read_text());value['metadata']['training_sources'][0]='E20'
    write_json(frozen['checkpoint'],value);frozen['freeze']['selected_checkpoint_sha256']=sha(frozen['checkpoint'])
    frozen['freeze']['training_sources']=value['metadata']['training_sources'];write_json(frozen['receipt'],frozen['freeze'])
    with pytest.raises(ValueError,match='training|overlap|membership'):
        m.evaluate_fidelity(frozen['checkpoint'],frozen['receipt'],frozen['holdout'],frozen['output'],v3_checkpoint=frozen['v3'])
