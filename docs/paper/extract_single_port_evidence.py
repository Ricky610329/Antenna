"""Freeze existing single-port handoff data; read-only NAS access, no HFSS imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path)
args = parser.parse_args()
root = args.dataset or Path(json.loads((BASE / 'final-candidate-response.json').read_text(encoding='utf-8'))['sources'][0]['path']).parent.parent
store = 'handoff_ant_db100'
pid = 'ANT_F00_1_c48nq1p06_18~db100'
original_store, original_id = 'dedust_r54valq01', 'c48nq1p06_18~db100'
input_path = root / (store + '_input') / (pid + '.pt')
bits = np.asarray(torch.load(input_path, weights_only=True)).reshape(25, 25)
assert np.isin(bits, [0, 1]).all()
cache = np.load(BASE.parents[1] / 'tmp/handoff_patterns.npz')
assert np.array_equal(bits, cache[f'{original_store}|{original_id}'])
assert np.array_equal(bits, cache[f'{store}|{pid}'])

matches = []
response_files = sorted((root / store).glob('*.pt'))
for path in response_files:
    x, y = torch.load(path, weights_only=True)
    if np.array_equal(np.asarray(x).reshape(25, 25), bits):
        matches.append((path, np.asarray(y, dtype=float)))
assert len(matches) == 1, f'Expected one exact input match, found {len(matches)}'
response_path, response = matches[0]
assert response.shape == (2, 17) and np.isfinite(response).all()
frequency = np.arange(24, 32.01, .5)
mask = np.arange(5, 12)
reflection_margin = float(-10 - response[0, mask].max())
realized_gain_margin = float(response[1, mask].min() - 4)
margin = min(reflection_margin, realized_gain_margin)

results_path = root / store / 'results.json'
record = json.loads(results_path.read_text(encoding='utf-8'))[pid]
assert [round(v, 2) for v in [reflection_margin, realized_gain_margin, margin]] == record['wm']
rad_path = root / store / 'rad' / (pid + '.pt')
rad = {k: np.asarray(v, dtype=float) for k, v in torch.load(rad_path, weights_only=True).items()}
theta = rad['theta']
assert theta.shape == (181,) and np.isfinite(theta).all()
assert np.array_equal(theta, np.arange(-180, 181, 2))
window = np.abs(theta) <= 45
zero = int(np.argmin(np.abs(theta)))
rad_margins = {k: float(rad[k][window].min() - (rad[k][zero] - 3)) for k in ['phi0', 'phi90']}
assert {k: round(v, 2) for k, v in rad_margins.items()} == record['rad']
assert round(min(rad_margins.values()), 2) == record['rad_margin']
original_results_path = root / original_store / 'results.json'
original_record = json.loads(original_results_path.read_text(encoding='utf-8'))[original_id]
setup_path = root / (store + '_input') / 'hfss_setup.json'
setup = json.loads(setup_path.read_text(encoding='utf-8'))
assert setup['diag_bridge_w'] == .1

def source(role, path):
    return {'role': role, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

frozen = {
    'schema_version': 1,
    'extracted_on': '2026-10-04',
    'scope': 'Existing single-port handoff confirmation data, not a new solve or a convergence claim.',
    'candidate': {'id': pid, 'store': store, 'original_id': original_id, 'original_store': original_store,
                  'bits': bits.astype(int).tolist(), 'pixel_pitch_mm': .2, 'metal_square_side_mm': .21,
                  'bridge_square_side_mm': .1, 'display_coordinates': 'u=HFSS Y, v=5mm-HFSS X; P1 below',
                  'match_method': 'Exact equality of all 625 input bits against formal input, response tuple, and both cached aliases',
                  'scanned_response_files': len(response_files), 'matched_response_files': len(matches)},
    'frequency_ghz': frequency.tolist(),
    'response_db': {'S11': response[0].tolist(), 'RealizedGainTotal': response[1].tolist()},
    'frequency_response_definition': {'gain': 'dB(RealizedGainTotal), theta=0 deg, phi=0 deg',
        'sweep': '24–32 GHz, 0.5 GHz exported frequency grid; existing Fast sweep output',
        'spec_sample_indices': mask.tolist(), 'spec_sample_ghz': frequency[mask].tolist(),
        'S11_upper_limit_db': -10, 'realized_gain_lower_limit_dbi': 4},
    'radiation': {'quantity': 'dB(GainTotal), not RealizedGainTotal', 'solution': 'Setup1 : LastAdaptive',
        'frequency_ghz': 28, 'theta_deg': theta.tolist(), 'phi0_db': rad['phi0'].tolist(), 'phi90_db': rad['phi90'].tolist(),
        'window_condition': 'abs(theta_deg) <= 45 on the saved 2-degree grid',
        'actual_window_endpoint_deg': [-44, 44], 'floor_below_each_cut_boresight_db': 3,
        'boresight_gain_dbi': {k: float(rad[k][zero]) for k in ['phi0', 'phi90']},
        'margin_db': rad_margins, 'worst_margin_db': min(rad_margins.values()),
        'display_note': 'Phi labels stay in the original HFSS coordinates; rotating the geometry drawing does not interchange cuts.'},
    'derived_check': {'reflection_margin_db': reflection_margin, 'realized_gain_margin_db': realized_gain_margin,
        'worst_inband_margin_db': margin, 'rounded_margins_db': [round(v, 2) for v in [reflection_margin, realized_gain_margin, margin]],
        'all_frequency_spec_samples_pass': bool(margin >= 0), 'radiation_window_samples_pass': bool(min(rad_margins.values()) >= 0)},
    'handoff_record': record,
    'original_batch_record': original_record,
    'original_vs_handoff': {'original_rounded_wm_db': original_record['wm'][2], 'handoff_rounded_wm_db': record['wm'][2],
        'rounded_difference_db': round(record['wm'][2] - original_record['wm'][2], 2),
        'interpretation': 'Same cached input bits, different existing evaluations. Plot and derived margins use the formal handoff evaluation together.'},
    'saved_setup_overrides': setup,
    'setup_limit': 'The saved override only records bridge, timeout and project retention; it is not a complete historical solver-settings archive.',
    'sources': [source('formal_input', input_path), source('matched_frequency_response', response_path),
                source('formal_radiation', rad_path), source('formal_results', results_path),
                source('original_batch_results', original_results_path), source('formal_setup_overrides', setup_path)],
    'code_sources': {'frequency_response': 'antenna/patch/patch_simulator/single_port.py:610–623',
        'radiation_response': 'antenna/patch/patch_simulator/single_port_rad.py:87–98',
        'radiation_metric': 'script/dedust.py:466–474'},
}
target = BASE / 'single-port-evidence.json'
target.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'file': str(target), 'wm': frozen['derived_check']['rounded_margins_db'],
                  'radiation': {k: round(v, 2) for k, v in rad_margins.items()},
                  'original_vs_handoff': frozen['original_vs_handoff']}, ensure_ascii=False))
