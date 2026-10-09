"""Plot a hash-bound current-profile symmetry analysis; no HFSS or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np

from script.figs.report_r1r10_style import polar_rad_ax


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


def render_sample(analysis: Path, npz: Path, out: Path, sample_id: str,
                  observation_note: str = '單次 HFSS 模擬；尚未重測驗證') -> dict:
    """Render one explicit, hash-bound profile observation without NAS discovery."""
    summary, data = _load(analysis, npz)
    indices = np.flatnonzero(data['ids'] == sample_id)
    if len(indices) != 1:
        raise ValueError('sample id must identify exactly one frozen observation')
    index = int(indices[0])
    row = summary['rows'][index]
    if row['id'] != sample_id or row['array_index'] != index:
        raise ValueError('sample JSON/NPZ row alignment differs')
    paths = [out / 'sample_card.png', out / 'sample_plot_receipt.json']
    if any(path.exists() for path in paths):
        raise FileExistsError('sample plot output already exists; choose a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    frequency = data['response_freqs_ghz']
    band = (frequency >= 26.5) & (frequency <= 29.5)
    s11, gain = data['responses'][index]
    s11_margin = float(-10 - s11[band].max())
    gain_margin = float(gain[band].min() - 4)
    wm = min(s11_margin, gain_margin)
    if (s11_margin != row['s11_band_margin_db']
            or gain_margin != row['gain_band_margin_db']):
        raise ValueError('sample plotted margins differ from validated analysis')
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), layout='constrained')
    geometry = axes[0, 0]
    geometry.imshow(data['patterns'][index].astype(bool), origin='upper',
                    cmap=ListedColormap([SURF, BLUE]), vmin=0, vmax=1,
                    interpolation='nearest')
    geometry.axvline(12, color=ORANGE, lw=.8, ls='--', alpha=.8)
    geometry.scatter([12], [24], marker='^', s=70, color=ORANGE, zorder=3)
    geometry.set(xticks=[], yticks=[], xlabel='25 × 25 金屬像素；橘色標記為下緣饋入位置',
                 title='金屬排列（像素俯視圖）')
    for ax, curve, threshold, name, unit, margin, worst in (
            (axes[0, 1], s11, -10, 'S11', 'dB', s11_margin, int(np.argmax(s11[band]))),
            (axes[0, 2], gain, 4, '正向 RealizedGainTotal', 'dBi', gain_margin, int(np.argmin(gain[band])))):
        ax.axvspan(26.5, 29.5, color=GRID, alpha=.35)
        ax.plot(frequency, curve, color=BLUE, lw=2)
        ax.plot([26.5, 29.5], [threshold, threshold], color=ORANGE,
                ls='--', label=f'帶內要求 {threshold:g} {unit}')
        point = np.flatnonzero(band)[worst]
        ax.scatter(frequency[point], curve[point], color=ORANGE, zorder=3)
        ax.annotate(f'{curve[point]:.5f} {unit}', (frequency[point], curve[point]),
                    xytext=(0, 12), textcoords='offset points', ha='center', fontsize=9)
        ax.set(xlabel='頻率（GHz）', ylabel=unit, xlim=(24, 32),
               title=f'{name}\n帶內餘裕 {margin:+.5f} dB')
        ax.grid(color=GRID, alpha=.65, lw=.7)
        ax.legend(loc='lower left', fontsize=9)
    info = axes[1, 0]
    info.axis('off')
    mismatch = float(data['geometry_mismatch_fraction'][index])
    info_text = info.text(.03, .96,
              f'WM = min(S11 餘裕, Gain 餘裕)\n\n'
              f'單次 WM = {wm:+.6f} dB\n'
              f'雙門檻：{"通過" if wm >= 0 else "尚未通過"}\n\n'
              f'金屬左右 mismatch = {mismatch:.6f}\n'
              f'φ=0° 鏡射功率殘差 = {row["phi0_mirror_power_45"]:.6f}\n'
              f'φ=90° 鏡射功率殘差 = {row["phi90_mirror_power_45"]:.6f}\n\n'
              '場型殘差計算範圍：±45°，0 越對稱\n'
              '帶內 spec：26.5–29.5 GHz\n'
              f'{observation_note}',
              va='top', fontsize=10, linespacing=1.25, color=INK)
    theta = data['radiation_theta_deg']
    if not np.array_equal(theta, -theta[::-1]):
        raise ValueError('sample plot requires the validated symmetric theta grid')
    cuts = [data[key][index] for key in ('phi0_db', 'phi90_db')]
    boresight = int(np.abs(theta).argmin())
    g0 = max(float(curve[boresight]) for curve in cuts)
    rmax = int(np.ceil((max(float(curve.max()) for curve in cuts) + .5) / 5) * 5)
    rmin = int(max(np.floor(min(float(curve.min()) for curve in cuts) / 5) * 5, rmax - 30))
    clipped = {}
    for column, key, phi in ((1, 'phi0_db', 0), (2, 'phi90_db', 90)):
        slot = axes[1, column].get_subplotspec()
        axes[1, column].remove()
        ax = fig.add_subplot(slot, projection='polar')
        curve = data[key][index]
        polar_rad_ax(ax, theta, [(curve, BLUE, 'HFSS 原始曲線', 2),
                                (curve[::-1], ORANGE, 'θ → −θ 鏡射曲線', 1.2)],
                     window=45, floor_db=3, rmin=rmin, rmax=rmax, g0_ref=g0)
        for line in ax.lines:
            if line.get_label() == 'θ → −θ 鏡射曲線':
                line.set_linestyle('--')
        ax.set_title(f'28 GHz 方向圖 φ={phi}°\nGainTotal（dBi）', fontsize=11, pad=16)
        ax.set_rlabel_position(15)
        ax.grid(color=GRID, alpha=.65, lw=.7)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, -.17), ncol=2,
                  fontsize=8, frameon=False)
        clipped[key] = int(np.count_nonzero(curve < rmin))
    info_text.set_text(info_text.get_text() + '\n\n'
                       '極座標沿用歷史圖：0°朝上、每圈5 dB\n'
                       '金色：±45°；紅虛圈：G0−3 dB\n'
                       f'顯示下限 {rmin} dBi；更深零點截至圓心')
    fig.suptitle(f'R80 金屬對稱樣本：單次 WM {wm:+.6f} dB\n{sample_id}', fontsize=15)
    fig.savefig(paths[0], dpi=160)
    plt.close(fig)
    receipt = {'schema_version': 1, 'sample_id': sample_id, 'pattern_sha256': row['pattern_sha256'],
               'analysis_sha256': _sha(analysis), 'npz_sha256': _sha(npz),
               'sample_sha256': row['raw_result']['sample_sha256'],
               'radiation_sha256': row['raw_result']['rad_sha256'],
               's11_margin_db': s11_margin, 'gain_margin_db': gain_margin, 'wm_db': wm,
               'geometry_mismatch_fraction': mismatch,
               'phi0_mirror_power_45': row['phi0_mirror_power_45'],
               'phi90_mirror_power_45': row['phi90_mirror_power_45'],
               'radiation_plot': {'projection': 'polar',
                                  'helper': 'script/figs/report_r1r10_style.py:polar_rad_ax',
                                  'helper_sha256': _sha(Path(__file__).with_name('report_r1r10_style.py')),
                                  'theta_zero': 'north', 'theta_direction': 'clockwise',
                                  'radial_unit': 'dBi', 'radial_tick_step_db': 5,
                                  'rmin_db': rmin, 'rmax_db': rmax, 'g0_ref_db': g0,
                                  'window_deg': 45, 'reference_drop_db': 3,
                                  'display_clipped_below_rmin': clipped},
               'producer_sha256': _sha(Path(__file__)), 'figure_sha256': _sha(paths[0]),
               'scope': 'one explicit frozen single HFSS solve; pixel view, frequency responses and GainTotal cuts; no repeat certification',
               'observation_note': observation_note}
    paths[1].write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-json', type=Path, required=True)
    parser.add_argument('--data-npz', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--sample-id', help='render one frozen sample card instead of population plots')
    parser.add_argument('--observation-note', default='單次 HFSS 模擬；尚未重測驗證',
                        help='explicit observation status shown on the sample card and saved in its receipt')
    args = parser.parse_args()
    if args.sample_id:
        result = render_sample(args.analysis_json, args.data_npz, args.out_dir, args.sample_id,
                               args.observation_note)
    else:
        result = render(args.analysis_json, args.data_npz, args.out_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
