"""Scientific plots and saved-record checks of spatial and modality tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .dataset import write_json
from .integrity import sha256_file
from .vla_evaluate import validate_manifest,validate_receipt


def analyze(base='outputs/vla_v5'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = Path(base)
    manifest,receipt,report = [json.loads((base/name).read_text(encoding='utf-8')) for name in
                               ('configurations.json','model_freeze.json','evaluation.json')]
    validate_receipt(receipt,manifest)
    if report['checkpoint_sha256'] != receipt['checkpoint_sha256'] or report['manifest_sha256'] != sha256_file(base/'configurations.json'):
        raise ValueError('Analysis inputs do not belong to frozen experiment')
    rows = report['rollouts']; output=base/'analysis'; output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    for ax,band in zip(axes,range(3)):
        grid=np.zeros((1,8),int)
        for direction in range(8):
            grid[0,direction]=sum(r['success'] for r in rows if r['suite']=='main' and r['distance_band_index']==band and r['direction_bin_index']==direction)
        ax.imshow(grid,vmin=0,vmax=4,cmap='Greens',aspect=1.3)
        for d in range(8): ax.text(d,0,f'{grid[0,d]}/4',ha='center',va='center',color='white')
        ax.set_yticks([]);ax.set_xticks(range(8));ax.set_xlabel('Direction bin')
        ax.set_title(('45-75 mm','75-110 mm','110-160 mm')[band])
    fig.suptitle('Frozen TinyVLA primary test: 96/96; novel combinations under a fixed camera')
    fig.savefig(output/'spatial_generalization.png',dpi=160);plt.close(fig)
    cfg=manifest['configurations'][0]
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    for task,col in ((0,'#276b51'),(1,'#4776bf'),(2,'#b77035')):
        row=next(r for r in rows if r['suite']=='language_counterfactual' and r['configuration_id']==cfg['configuration_id'] and r['instruction_index']==task)
        xy=np.array([s['ring_xyz_m'][:2] for s in row['trajectory']])*1000
        ax.plot(xy[:,0],xy[:,1],color=col,label=('inside','left','right')[task],lw=2)
        ax.scatter(*xy[-1],color=col,s=40)
    ax.set(xlabel='Actual ring X (mm)',ylabel='Actual ring Y (mm)',title='Same initial RGB, different text: actual physical ring paths')
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'language_physical_paths.png',dpi=160);plt.close(fig)
    mean_errors={};heights={}
    for suite in ('main','black_image','language_counterfactual'):
        group=[r for r in rows if r['suite']==suite]
        errors=[]
        for r in group:
            last=r['trajectory'][-1]
            goal=np.array(last['bearing_xyz_m'][:2])+[(-.06 if r['instruction_index']==1 else .06 if r['instruction_index']==2 else 0.),0.]
            errors.append(np.linalg.norm(np.array(last['ring_xyz_m'][:2])-goal))
        mean_errors[suite]=dict(mean_m=float(np.mean(errors)),max_m=float(np.max(errors)))
        heights[suite]=dict(minimum_case_max_height_m=min(max(s['ring_xyz_m'][2] for s in r['trajectory']) for r in group))
    result=dict(status='passed',checkpoint_sha256=receipt['checkpoint_sha256'],evaluation_sha256=sha256_file(base/'evaluation.json'),
        attempts=len(rows),all_completed=all(r['controller_finished'] for r in rows),
        forbidden_accesses=sum(r['policy_forbidden_access_attempts'] for r in rows),
        final_center_errors=mean_errors,height_checks=heights,summary=report['summary']['by_suite'],
        paired_diagnostics=report['paired_diagnostics'],
        scope='Fixed robot-hidden simulator RGB, finite language, same objects; waypoint execution manual; no real-camera claims')
    if len(rows)!=264 or result['forbidden_accesses'] or not result['all_completed']:
        raise ValueError('Incomplete or invalid saved experiment')
    write_json(output/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('paired_diagnostics','height_checks')},indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',default='outputs/vla_v5')
    analyze(parser.parse_args().base)
