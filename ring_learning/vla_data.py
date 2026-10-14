"""Source-isolated synthetic RGB/action chunks retargeted from recorded motion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import split_episodes, write_json
from .generalization_policy import load_generalization_policy
from .integrity import sha256_file
from .vla_env import VLAEnv, camera_signature


TASK_NAMES = ('inside','left','right')


def action_chunk(ring_xy,bearing_xy,task,shape):
    if task not in TASK_NAMES:
        raise ValueError('Unknown training task')
    start = np.asarray(ring_xy,float)+[0.,.01125]
    goal = np.asarray(bearing_xy,float)+[dict(inside=0.,left=-.06,right=.06)[task],.01125]
    sequence = [np.r_[start,.1693,1.],np.r_[start,.1043,1.],
                np.r_[start,.1043,-1.],np.r_[start,.1643,-1.]]
    for progress in np.linspace(1/16,1,16):
        sequence.append(np.r_[shape.waypoint(start,goal,float(progress)),.1643,-1.])
    sequence.extend([np.r_[goal,.1043,-1.],np.r_[goal,.1043,1.],np.r_[goal,.1943,1.]])
    return np.array(sequence,np.float32)


def generate(output='outputs/vla_v5/dataset',seed=20261025):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new data destination')
    output.mkdir(parents=True)
    real = json.loads(Path('data/real_episodes.json').read_text(encoding='utf-8'))
    splits = split_episodes(real['episodes'])
    teacher_path = 'outputs/generalization_v4/model/policy.json'
    shape,metadata = load_generalization_policy(teacher_path)
    if set(metadata['training_sources']) != {e['episode_id'] for e in splits['train']}:
        raise ValueError('Optical motion teacher is not restricted to original training sources')
    rng = np.random.default_rng(seed)
    env, configs, files = VLAEnv(), {}, {}
    try:
        for split, count in (('train',16),('val',8)):
            images,masks,coords,chunks,source_ids,identifiers = [],[],[],[],[],[]
            configs[split] = []
            for ep in sorted(splits[split],key=lambda e:e['episode_id']):
                length = float(np.linalg.norm(np.array(ep['bearing_xy_m'])-ep['ring_xy_m']))
                for draw in range(count):
                    for _ in range(1000):
                        angle = float(rng.uniform(-np.pi,np.pi))
                        delta = length*np.array([np.cos(angle),np.sin(angle)])
                        low = np.maximum([.40,-.10],np.array([.30,-.25])+delta)
                        high = np.minimum([.52,.10],np.array([.64,.25])+delta)
                        if np.all(low <= high):
                            break
                    else:
                        raise ValueError('Source cannot fit declared training workspace')
                    bearing = rng.uniform(low,high)
                    ring = bearing-delta
                    cfg = dict(configuration_id=f'V5_{ep["episode_id"]}_{split}_{draw:03d}',
                        source_episode_id=ep['episode_id'],split=split,ring_xy_m=ring.tolist(),
                        bearing_xy_m=bearing.tolist(),rotation_draw_rad=angle,
                        source_endpoint_distance_m=length,reset_seed=int(rng.integers(2**31-1)))
                    env.reset(ring,bearing,cfg['reset_seed'])
                    images.append(env.render_policy())
                    masks.append(env.render_masks())
                    coords.append(np.r_[ring,bearing])
                    chunks.append([action_chunk(ring,bearing,task,shape) for task in TASK_NAMES])
                    source_ids.append(ep['episode_id']); identifiers.append(cfg['configuration_id'])
                    configs[split].append(cfg)
                print(f'{split}/{ep["episode_id"]}: {len(images)} images',flush=True)
            path = output/f'{split}.npz'
            np.savez_compressed(path,images=np.asarray(images,np.uint8),masks=np.asarray(masks,bool),
                coordinates=np.asarray(coords,np.float32),waypoints=np.asarray(chunks,np.float32),
                source_episode_ids=np.asarray(source_ids),configuration_ids=np.asarray(identifiers))
            files[path.name] = dict(sha256=sha256_file(path),images=len(images),bytes=path.stat().st_size)
    finally:
        env.close()
    manifest = dict(schema_version=1,seed=seed,task_names=list(TASK_NAMES),configurations=configs,files=files,
        split_sources={s:sorted(e['episode_id'] for e in splits[s]) for s in ('train','val')},
        source_video_sha256=real['source']['sha256'],real_episodes_sha256=sha256_file('data/real_episodes.json'),
        development_annotations_sha256=sha256_file('data/v4_development_annotations.json'),
        video_teacher=dict(checkpoint=teacher_path,sha256=sha256_file(teacher_path),
            training_sources=metadata['training_sources'],description='Frozen train-only real optical motion model; '
            'robot height/grip and left/right instruction labels are synthetic, not observed phone actions'),
        camera=camera_signature(),camera_convention='Static robot-hidden full-tabletop RGB pass; training masks never runtime inputs',
        source_isolation=True,test_data_generated=False,
        code_sources={name:hashlib.sha256(Path('ring_learning',name+'.py').read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest()
                      for name in ('vla_data','vla_env')},
        supervision='Rendered initial RGB/masks and23 absolute XYZ/grip targets; not measured robot demonstrations',
        training_workspace=dict(ring_lower=[.30,-.25],ring_upper=[.64,.25],bearing_lower=[.40,-.10],bearing_upper=[.52,.10]))
    write_json(output/'manifest.json',manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/vla_v5/dataset')
    parser.add_argument('--seed',type=int,default=20261025)
    args = parser.parse_args()
    generate(args.output,args.seed)
