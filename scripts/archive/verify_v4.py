"""Verify frozen bindings and all saved physical/fidelity records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ring_learning.generalization_evaluate import validate_manifest
from ring_learning.generalization_fidelity import _validate_freeze
from ring_learning.integrity import sha256_file


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify():
    base = Path('outputs/generalization_v4')
    manifest = read(base/'configurations.json')
    validate_manifest(manifest)
    model, _, receipt, _, _, provenance = _validate_freeze(base/'model/policy.json',
        base/'model_freeze.json', 'data/v4_holdout_annotations.json', None)
    rows, physical, fidelity = read(base/'evaluation/rollouts.json'), read(base/'evaluation/results.json'), read(base/'fidelity/fidelity.json')
    expected = {(name,c['configuration_id']) for name in ('v4','straight','v3_profile') for c in manifest['configurations']}
    actual = {(r['controller'],r['configuration_id']) for r in rows}
    if len(rows) != 288 or actual != expected:
        raise ValueError('Saved evaluation omits or duplicates a predeclared attempt')
    if physical['manifest_sha256'] != sha256_file(base/'configurations.json') or physical['selected_checkpoint_sha256'] != receipt['selected_checkpoint_sha256']:
        raise ValueError('Physical results are not bound to the frozen model and scenes')
    if fidelity['provenance'] != provenance:
        raise ValueError('Fidelity source/data hashes differ from current saved artifacts')
    if (fidelity['observed_points'],fidelity['intermediate_points']) != (35,29):
        raise ValueError('Locked fidelity point counts differ')
    maximum_transfer_displacement_m, early_lowerings = 0., 0
    for row in rows:
        transferred = [s for s in row['trajectory'] if s['phase_before']==4 and s['action'] is not None]
        for sample in transferred:
            displacement = float(np.linalg.norm(np.asarray(sample['action'][:3])*.003))
            maximum_transfer_displacement_m = max(maximum_transfer_displacement_m,displacement)
            early_lowerings += int(sample['phase_after']==5 and sample['progress'] < 1.-1e-12)
        if row['transfer_contact_steps'] != len(transferred) or row['reference_final_progress'] != 1.:
            raise ValueError('A saved case lost transfer contact or did not complete the reference')
        expected_calls = 0 if row['controller']=='straight' else 63
        if row['learned_reference_evaluations'] != expected_calls:
            raise ValueError('Saved inference counts differ from actual reference construction')
        states = np.array([s['ring_xyz_m']+s['tool_xyz_m']+s['command_xyz_m'] for s in row['trajectory']])
        if not np.isfinite(states).all():
            raise ValueError('Saved physical state contains nonfinite coordinates')
    if early_lowerings or maximum_transfer_displacement_m > .002000001:
        raise ValueError('Transfer speed/completion constraints violated in saved output')
    summaries = physical['summaries']['overall']
    for name in ('v4','straight','v3_profile'):
        group = [r for r in rows if r['controller']==name]
        if summaries[name]['attempted_cases'] != 96 or summaries[name]['successes'] != sum(r['success'] for r in group):
            raise ValueError('Published summary disagrees with all attempted outcomes')
    result = dict(status='passed', model_parameters=model.parameter_count, physical_attempts=len(rows),
        successes={name:summaries[name]['successes'] for name in summaries},
        maximum_transfer_action_displacement_m=maximum_transfer_displacement_m, early_lowerings=early_lowerings,
        physical_contact_preserved_in_all_transfer_steps=True, provenance=provenance,
        files={str(path):sha256_file(path) for path in (base/'configurations.json',base/'model/policy.json',
            base/'model_freeze.json',base/'evaluation/results.json',base/'evaluation/rollouts.json',base/'fidelity/fidelity.json')})
    destination = base/'final_verification.json'
    destination.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','provenance')},indent=2))
    return result


if __name__ == '__main__':
    verify()
