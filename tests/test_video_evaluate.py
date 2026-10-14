import importlib
import json

import pytest


def test_frozen_video_scenes_are_source_isolated_and_cannot_be_overwritten(tmp_path):
    module = importlib.import_module('ring_learning.video_evaluate')
    destination = tmp_path / 'configurations.json'
    manifest = module.prepare('data/real_episodes.json', destination)
    groups = manifest['configurations']
    assert len(groups['test']) == 18
    assert len(groups['ood_test']) == 30
    sources = {name: {c['source_episode_id'] for c in cases} for name, cases in groups.items()}
    for name, ids in sources.items():
        for other, other_ids in sources.items():
            if name != other:
                assert not ids & other_ids
    first_bytes = destination.read_bytes()
    with pytest.raises(FileExistsError):
        module.prepare('data/real_episodes.json', destination)
    assert destination.read_bytes() == first_bytes
    assert json.loads(first_bytes)['protocol'].startswith('video_motion_v3')


def test_video_report_counts_failures_and_planned_steps():
    module = importlib.import_module('ring_learning.video_evaluate')
    rows = [dict(controller='learned_motion', split='test', success=True,
                 source_episode_id='E01', learned_steps=40, planned_steps=100, info={}),
            dict(controller='learned_motion', split='test', success=False,
                 source_episode_id='E02', learned_steps=0, planned_steps=450,
                 info={'failure_reason': 'timeout_approach'})]
    result = module.summarize(rows)['test/learned_motion']
    assert result['successes'] == 1
    assert result['episodes'] == 2
    assert result['success_rate'] == .5
    assert result['planned_steps'] == 550
    assert result['learned_steps'] == 40
    assert result['failure_counts'] == {'timeout_approach': 1}


def test_model_cannot_use_stale_endpoint_manifest_with_same_source_ids():
    module = importlib.import_module('ring_learning.video_evaluate')
    manifest = {'real_episodes_sha256': 'a'*64, 'configurations': {
        'train': [{'source_episode_id': 'E01'}], 'test': [{'source_episode_id': 'E02'}]}}
    metadata = {'real_episodes_sha256': 'b'*64, 'training_sources': ['E01']}
    with pytest.raises(ValueError, match='endpoint manifest'):
        module.validate_model_provenance(metadata, manifest, ['test'])
    metadata['real_episodes_sha256'] = 'a'*64
    module.validate_model_provenance(metadata, manifest, ['test'])
