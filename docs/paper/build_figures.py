"""Rebuild paper figures from the adjacent frozen evidence data; never runs HFSS."""
import json
import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

BASE = Path(__file__).resolve().parent
data = json.loads((BASE / 'paper-evidence-data.json').read_text(encoding='utf-8'))
out = BASE / 'figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'svg.hashsalt': 'antenna-paper-2026-10'})
teal, amber, ink = '#087d80', '#bd6532', '#1e3346'

c = data['candidate']
bits = np.asarray(c['bits'])
assert bits.shape == (25, 25) and np.isin(bits, [0, 1]).all()
margins = [round(c['m'][0]+2, 2), round(c['m'][1]+2, 2), c['m'][2], round(c['m'][3]+5, 2)]
assert margins == c['spec_v2_margins'] and min(margins) == c['score']
assert len(data['history']) == 17 and np.all(np.diff([r['value'] for r in data['history']]) >= 0)
assert len(data['online_r73']) == 15 and sum(r['n'] for r in data['online_r73']) == 135
raw = json.loads((BASE / 'final-candidate-response.json').read_text(encoding='utf-8'))
f = np.asarray(raw['frequency_ghz'])
s11, s21, s22 = [np.asarray(raw['response_db'][label]) for label in ['S11','S21','S22']]
assert np.array_equal(f, np.arange(24, 32.01, .5))
assert all(len(a)==17 and np.isfinite(a).all() for a in (s11, s21, s22))
derived = [-10-s11[5:12].max(), -10-s22[5:12].max(), s21[3:14].min()+3, -15-np.r_[s21[:3], s21[14:]].max()]
assert [round(float(a),2) for a in derived] == margins

# Load only the simulator's pure geometry function, without importing the simulator
# package, creating a data store, loading a network, or opening HFSS.
geom_path = BASE.parents[1] / 'antenna/patch/patch_simulator/single_port.py'
tree = ast.parse(geom_path.read_text(encoding='utf-8'))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'diag_bridge_sites')
namespace = {'np': np}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(geom_path), 'exec'), namespace)
sites, skipped = namespace['diag_bridge_sites'](bits, .075, .2)
assert len(sites) == 67 and skipped == 0

fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.25), gridspec_kw={'width_ratios':[1,1.25,1.25]}, layout='constrained')
for row, col in zip(*np.nonzero(bits)):
    axes[0].add_patch(Rectangle((row*.2, col*.2), .21, .21, facecolor=ink, edgecolor='none'))
for sx, sy, width in sites:
    delta = width / np.sqrt(2)
    axes[0].add_patch(Polygon([(sx-delta,sy),(sx,sy+delta),(sx+delta,sy),(sx,sy-delta)], closed=True, facecolor=amber, edgecolor='none'))
axes[0].set(xlim=(-.18,5.18), ylim=(-.18,5.18), xlabel='HFSS x (mm)', ylabel='HFSS y (mm)', title='(a) Pixel-region geometry')
axes[0].set_aspect('equal')
for px, marker, label, text_x in [(0,'>', 'P2', .18), (5,'<','P1',4.5)]:
    axes[0].scatter([px],[2.5],marker=marker,s=65,c=teal,edgecolor='white',linewidth=.7,zorder=10)
    axes[0].text(text_x,2.82,label,fontsize=8,color=teal,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1})
axes[0].text(.5,-.27, '67 bridges (orange), width 0.075 mm\nP1/P2: schematic feed connections\nExternal feeds / substrate / ground omitted', transform=axes[0].transAxes, ha='center', fontsize=8)

axes[1].axvspan(26.5,29.5, color=teal, alpha=.10)
axes[1].plot(f, s11, 'o-', color=teal, ms=3, label='S11')
axes[1].plot(f, s22, 's-', color=ink, ms=3, label='S22')
axes[1].hlines(-10,26.5,29.5, color=amber, ls='--', lw=1.6, label='Reflection limit')
axes[1].set(title='(b) Reflection', xlabel='Frequency (GHz)', ylabel='S-parameter (dB)', xlim=(24,32), ylim=(-31,0), xticks=[24,26,28,30,32])
axes[1].legend(loc='lower left', fontsize=8, frameon=False)
axes[1].text(.38,.94, f'Margins:\n{margins[0]:+.2f} / {margins[1]:+.2f} dB', transform=axes[1].transAxes, va='top', fontsize=8)

axes[2].axvspan(25.5,30.5, color=teal, alpha=.10)
for lo,hi in [(24,25),(31,32)]:
    axes[2].axvspan(lo,hi, color=amber, alpha=.09)
    axes[2].hlines(-15,lo,hi,color=amber,ls='--',lw=1.6)
axes[2].hlines(-3,25.5,30.5,color=amber,ls='--',lw=1.6, label='Spec limits')
axes[2].plot(f,s21,'o-',color=teal,ms=3,label='S21')
bad = np.zeros(17,dtype=bool)
bad[3:14] = s21[3:14] < -3
bad[:3] = s21[:3] > -15
bad[14:] = s21[14:] > -15
assert bad.sum() == 6
axes[2].scatter(f[bad],s21[bad],s=45,facecolors='none',edgecolors=amber,zorder=5,label='Unmet sampled point')
axes[2].set(title='(c) Transmission', xlabel='Frequency (GHz)', ylabel='S21 (dB)', xlim=(24,32), ylim=(-31,0), xticks=[24,26,28,30,32])
axes[2].legend(loc='lower center',fontsize=8,frameon=False)
axes[2].text(.28,.57, f'Pass / stop margins:\n{margins[2]:+.2f} / {margins[3]:+.2f} dB',transform=axes[2].transAxes,fontsize=8)
for ax in axes[1:]:
    ax.grid(alpha=.18)
fig.savefig(out / 'candidate.svg', metadata={'Date': None})
fig.savefig(out / 'candidate.png', dpi=165)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout='constrained')
hist = data['history']
x = np.arange(1, len(hist)+1)
vals = [r['value'] for r in hist]
axes[0].step(x, vals, where='post', color=teal, lw=2)
axes[0].scatter(x, vals, color=teal, s=20, zorder=3)
axes[0].axhline(0, color=ink, ls='--', lw=1, label='All four margins pass')
axes[0].set(xlabel='Recorded best-so-far milestone (not query count)', ylabel='Worst specification margin (dB)', title='(a) Fixed profile: p01 + 0.075 mm bridges', ylim=(-6.4, .5), xticks=[1, 5, 9, 13, 17])
axes[0].annotate('-5.90', (1, vals[0]), xytext=(6, 9), textcoords='offset points')
axes[0].annotate('-2.39', (17, vals[-1]), xytext=(-34, 12), textcoords='offset points')
axes[0].text(.03, .77, '2026-08-13 to 08-24\nBest-so-far change: 3.51 dB', transform=axes[0].transAxes, fontsize=9)
axes[0].grid(alpha=.18)
r73 = data['online_r73']
rounds = [r['iter'] for r in r73]
v = [r['wm'] for r in r73]
cum = np.maximum.accumulate([-3.15] + v)[1:]
axes[1].plot(rounds, v, 'o-', color=amber, ms=4, label='Best returned in each batch')
axes[1].step(rounds, cum, where='post', color=teal, lw=2, label='Best-so-far incl. starting record')
axes[1].set(xlabel='R73 iteration (9 HFSS results per iteration)', ylabel='Worst specification margin (dB)', title='(b) Multi-candidate online-loop case', xticks=[1, 3, 6, 9, 12, 15], ylim=(-10.3, -2.2))
axes[1].grid(alpha=.18)
axes[1].legend(loc='lower left', fontsize=8, frameon=False)
axes[1].annotate('Round 7: -3.14 dB', (7, -3.14), xytext=(7.4, -4.9), arrowprops={'arrowstyle':'-', 'color': ink}, fontsize=9)
fig.savefig(out / 'search-history.svg', metadata={'Date': None})
fig.savefig(out / 'search-history.png', dpi=165)
plt.close(fig)
for svg in out.glob('*.svg'):
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
print('Wrote two SVG/PNG figure pairs from frozen data.')
