"""Plot a hash-bound current-profile symmetry analysis; no HFSS or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


INK, MUTED, GRID, SURF = '#171716', '#66645f', '#dedcd4', '#fcfcfb'
BLUE, ORANGE = '#1c5cab', '#e36a32'
EXTENT = (0.0, 1.0, -1.0, 1.0)
GRIDSIZE = (24, 24)
plt.rcParams.update({'font.family': ['Microsoft JhengHei', 'sans-serif'],
                     'axes.unicode_minus': False, 'font.size': 10,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF})


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load(analysis: Path, npz: Path):
    summary = json.loads(analysis.read_text(encoding='utf-8'))
    if _sha(npz) != summary['artifacts']['npz']['sha256']:
        raise ValueError('analysis/NPZ hash binding differs')
    with np.load(npz, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    n = len(data['ids'])
    if not n or len(set(data['pattern_sha256'].tolist())) != n:
        raise ValueError('plots require nonempty unique physical patterns')
    if summary['counts']['unique_selected_rows'] != n:
        raise ValueError('analysis count differs from NPZ rows')
    if data['responses'].shape != (n, 2, 17):
        raise ValueError('plots require the current single-port 17-frequency profile')
    if not np.array_equal(data['response_freqs_ghz'], np.linspace(24, 32, 17)):
        raise ValueError('plots require the fixed 24..32 GHz half-GHz grid')
    for key in ('metal_fraction', 'upper_lower_metal_imbalance', 's11_band_margin_db',
                'gain_band_margin_db', 'gain_db_28', 'phi0_mirror_power_45',
                'phi90_mirror_power_45'):
        if data[key].shape != (n,) or not np.isfinite(data[key]).all():
            raise ValueError(f'invalid plotting field: {key}')
    if not np.isfinite(data['responses']).all():
        raise ValueError('nonfinite frequency response')
    return summary, data


def terrain(data: dict, out: Path) -> None:
    x, y = data['metal_fraction'], data['upper_lower_metal_imbalance']
    panels = [
        ('s11_band_margin_db', '帶內 S11 餘裕', 'dB（0 以上達舊門檻）', 'Blues', -15, 0),
        ('gain_band_margin_db', '帶內 Gain 餘裕', 'dB（0 以上達舊門檻）', 'Blues', -45, 0),
        ('gain_db_28', '28 GHz 正向 RealizedGainTotal', 'dBi（4 以上深色）', 'Blues', -25, 4),
        ('phi0_mirror_power_45', 'φ=0° 場型鏡射殘差（±45°）', '線性功率殘差（0 較對稱）', 'Blues_r', 0, 1),
        ('phi90_mirror_power_45', 'φ=90° 場型鏡射殘差（±45°）', '線性功率殘差（0 較對稱）', 'Blues_r', 0, 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 10.2), layout='constrained')
    for ax, panel in zip(axes.flat, panels):
        key, title, label, cmap, vmin, vmax = panel
        hb = ax.hexbin(x, y, C=data[key], reduce_C_function=np.median,
                       mincnt=1, gridsize=GRIDSIZE, extent=EXTENT,
                       cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        fig.colorbar(hb, ax=ax, shrink=.84, pad=.025, extend='both').set_label(label, fontsize=9)
        ax.set_title(title, fontsize=12, pad=10)
    density = axes.flat[-1]
    hb = density.hexbin(x, y, mincnt=1, gridsize=GRIDSIZE, extent=EXTENT,
                        cmap='Greys', linewidths=0)
    maximum = max(2.0, float(hb.get_array().max()))
    hb.set_norm(LogNorm(vmin=1, vmax=maximum))
    cb = fig.colorbar(hb, ax=density, shrink=.84, pad=.025)
    cb.set_label('每格唯一實測數（log）', fontsize=9)
    density.set_title('資料密度', fontsize=12, pad=10)
    for ax in axes.flat:
        ax.patch.set_hatch('///')
        ax.patch.set_edgecolor('#e5e3df')
        ax.set(xlim=EXTENT[:2], ylim=EXTENT[2:],
               xlabel='金屬面積比例', ylabel='上半部 − 下半部金屬比例')
        ax.axhline(0, color=GRID, lw=.7, zorder=1)
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    n = len(data['ids'])
    exact = int(np.count_nonzero(data['geometry_mismatch_fraction'] == 0))
    fig.suptitle(f'R80 金屬對稱資料：幾何分布與實測指標（n={n:,}；精確鏡射 {exact:,}）\n'
                 '座標僅由幾何定義；六角格=中位數，斜線=沒有樣本；指標深色=較好／較對稱，密度深色=較多',
                 color=INK, fontsize=14)
    fig.supxlabel('單次 HFSS；帶內為 26.5–29.5 GHz。場型為 28 GHz GainTotal；尚無配對非對稱對照，不作因果推論。',
                  color=MUTED, fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def frequency_curves(data: dict, out: Path) -> None:
    n = len(data['ids'])
    frequencies = data['response_freqs_ghz']
    traces = np.unique(np.linspace(0, n - 1, min(200, n), dtype=int))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), layout='constrained')
    panels = [(data['s11_db'], 'S11', 'dB', -10),
              (data['gain_db'], '正向 RealizedGainTotal', 'dBi', 4)]
    for ax, (values, title, unit, threshold) in zip(axes, panels):
        ax.axvspan(26.5, 29.5, color='#f0e6d9', alpha=.55, zorder=0)
        ax.plot(frequencies, values[traces].T, color='#91a5ba', alpha=.12, lw=.7)
        lo, middle, hi = np.quantile(values, [.25, .5, .75], axis=0)
        ax.fill_between(frequencies, lo, hi, color=BLUE, alpha=.22, label='全部樣本的 25–75%')
        ax.plot(frequencies, middle, color=BLUE, lw=2.2, label='全部樣本的中位數')
        ax.plot([26.5, 29.5], [threshold, threshold], color=ORANGE, ls='--', lw=1.6,
                label=f'既有帶內門檻 {threshold} {unit}')
        ax.set(xlabel='頻率（GHz）', ylabel=unit, title=title, xlim=(24, 32))
        ax.grid(color=GRID, lw=.6, alpha=.6)
        ax.legend(fontsize=8.5, frameon=False, loc='best')
        for spine in ax.spines.values():
            spine.set_color(GRID)
    fig.suptitle(f'R80 固定金屬鏡射條件下的頻率響應（{n:,} 個唯一圖形）', fontsize=14, color=INK)
    fig.supxlabel(f'灰線為固定排序取出的 {len(traces)} 筆單次 HFSS；區間是逐頻率分布，不是信賴區間或單一可製作圖形。',
                  fontsize=10, color=MUTED)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render(analysis: Path, npz: Path, out: Path) -> dict:
    summary, data = _load(analysis, npz)
    paths = [out / 'geometry_terrain.png', out / 'frequency_responses.png', out / 'plot_receipt.json']
    if any(p.exists() for p in paths):
        raise FileExistsError('plot output already exists; preserve it and choose a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    terrain(data, paths[0])
    frequency_curves(data, paths[1])
    receipt = {'schema_version': 1, 'analysis_path': str(analysis.resolve()),
               'analysis_sha256': _sha(analysis), 'npz_path': str(npz.resolve()), 'npz_sha256': _sha(npz),
               'unique_patterns': len(data['ids']), 'geometry_extent': list(EXTENT),
               'hex_gridsize': list(GRIDSIZE), 'hex_reduction': 'median, mincnt=1; no interpolation',
               'coordinate_policy': 'fixed geometric metal fraction and upper-minus-lower fraction; no response embedding',
               'frequency_quantiles': [0.25, 0.5, 0.75], 'scope': summary.get('scope'),
               'figures': {p.name: _sha(p) for p in paths[:2]}}
    paths[2].write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-json', type=Path, required=True)
    parser.add_argument('--data-npz', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.analysis_json, args.data_npz, args.out_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
