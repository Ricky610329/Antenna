"""Saved-profile polar delivery using the repository's existing helper."""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from script.figs.symmetry_profile import _load, plt, np
from script.figs.report_r1r10_style import polar_rad_ax

SOURCE = Path(r"\\140.123.106.219\temp\碩二_鄒穎麒's\antenna\experiments\r80_symmetry_20261007\analysis_versions\final5000_20261011_v001\analysis")
OUT = REPO / "docs/log/assets/round-80/final5000-20261011-v001"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary, data = _load(SOURCE / 'combined.json', SOURCE / 'combined.npz')
assert sha(SOURCE / 'combined.json') == '188f4fe03759358ded93224551c18cee96e02f1bd1723d6e8e1de3a7896f2642'
assert sha(SOURCE / 'combined.npz') == '11033babc2fa5c3d2303b0fa87e9054b6c725c1fee91cec940121d6716ab63bd'
theta = data['radiation_theta_deg']
curves = data['radiation']
assert curves.shape == (5000, 2, 181) and np.isfinite(curves).all()
assert np.array_equal(theta, np.linspace(-180, 180, 181))
assert np.count_nonzero(data['geometry_mismatch_fraction']) == 0
matches = np.flatnonzero(data['ids'] == 'r80localv1_00026_f6ffb938')
assert len(matches) == 1
best = int(matches[0])
assert data['pattern_sha256'][best] == 'f6ffb93885baa25fdcf82e06abf9a564bc27285f5b6202e3828871b605e4f784'
image = OUT / 'radiation_population_polar.png'
receipt = OUT / 'polar_plot_receipt.json'
code = OUT / 'render_population_polar.py'
assert not any(p.exists() for p in (image, receipt, code))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.2), subplot_kw={'projection': 'polar'})
for cut, ax in enumerate(axes):
    q25, median, q75 = np.quantile(curves[:, cut], [.25, .5, .75], axis=0)
    series = [(q25, '#90a9c2', '全體 25% 分位數', 1.2),
              (q75, '#90a9c2', '全體 75% 分位數', 1.2),
              (median, '#1c5cab', '全體中位數', 2.1),
              (curves[best, cut], '#e36a32', '最佳 WM 圖形（原次）', 2.1)]
    polar_rad_ax(ax, theta, series, rmin=-25, rmax=10,
                 g0_ref=float(curves[best, cut, 90]))
    ax.set_title(f'28 GHz GainTotal：φ={cut * 90}°', pad=24)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.26), frameon=False, fontsize=9)
fig.suptitle('R80：5,000 個唯一金屬左右對稱圖形的場型分布', fontsize=15, y=.97)
fig.text(.5, .065, '逐角度分位數僅描述資料分布，不是單一圖形；未做鏡射平均或插值。半徑為 dBi，低於 −25 僅在圖上截限。', ha='center', fontsize=10)
fig.text(.5, .035, '金色區域為 ±45° 軟性觀察窗；紅虛線是各切面最佳圖形正向 GainTotal −3 dB，皆不參與 WM。', ha='center', fontsize=10)
fig.subplots_adjust(top=.83, bottom=.27, wspace=.35)
fig.savefig(image, dpi=150)
plt.close(fig)
with code.open('xb') as stream:
    stream.write(Path(__file__).read_bytes())
result = {'schema_version': 1, 'unique_patterns': 5000,
          'source_json_sha256': sha(SOURCE / 'combined.json'),
          'source_npz_sha256': sha(SOURCE / 'combined.npz'),
          'best_id': str(data['ids'][best]), 'projection': 'polar',
          'helper': 'script.figs.report_r1r10_style.polar_rad_ax',
          'helper_sha256': sha(REPO / 'script/figs/report_r1r10_style.py'),
          'loader_sha256': sha(REPO / 'script/figs/symmetry_profile.py'),
          'quantiles': [.25, .5, .75], 'theta_deg': [-180, 180, 2],
          'quantity': '28 GHz LastAdaptive GainTotal', 'radial_limits_db': [-25, 10],
          'interpolation': False, 'mirror_averaging': False,
          'hfss_or_model_training': False,
          'figure_sha256': sha(image), 'code_sha256': sha(code)}
with receipt.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(result, stream, indent=2, ensure_ascii=False)
    stream.write('\n')
print(json.dumps(result, ensure_ascii=False))
