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
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'svg.hashsalt': 'antenna-paper-2026-10'})
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

def save_figure(fig, basename):
    """Save only the new manuscript assets; leave archived composite figures intact."""
    svg = out / f'{basename}.svg'
    fig.savefig(svg, metadata={'Date': None})
    fig.savefig(out / f'{basename}.png', dpi=165)
    plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')


fig, ax = plt.subplots(figsize=(6.4, 6.6), layout='constrained')
for row, col in zip(*np.nonzero(bits)):
    ax.add_patch(Rectangle((row*.2, col*.2), .21, .21, facecolor=ink, edgecolor='none'))
for sx, sy, width in sites:
    delta = width / np.sqrt(2)
    ax.add_patch(Polygon([(sx-delta,sy),(sx,sy+delta),(sx+delta,sy),(sx,sy-delta)], closed=True, facecolor=amber, edgecolor='none'))
ax.set(xlim=(-.18,5.18), ylim=(-.18,5.18), xlabel='HFSS x (mm)', ylabel='HFSS y (mm)', title='Record candidate: pixel-region geometry')
ax.set_aspect('equal')
for px, marker, label, text_x in [(0,'>', 'P2', .18), (5,'<','P1',4.5)]:
    ax.scatter([px],[2.5],marker=marker,s=80,c=teal,edgecolor='white',linewidth=.8,zorder=10)
    ax.text(text_x,2.82,label,fontsize=11,color=teal,bbox={'facecolor':'white','edgecolor':'none','alpha':.95,'pad':1})
fig.supxlabel('67 orange bridges: square side 0.075 mm\np01 overlap 0.01 mm; P1/P2: schematic feed connections\nExternal feeds / substrate / ground omitted', fontsize=10)
save_figure(fig, 'candidate-geometry')

fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), layout='constrained')
axes[0].axvspan(26.5,29.5, color=teal, alpha=.10)
axes[0].plot(f, s11, 'o-', color=teal, ms=4, label='S11')
axes[0].plot(f, s22, 's-', color=ink, ms=4, label='S22')
axes[0].hlines(-10,26.5,29.5, color=amber, ls='--', lw=1.6)
axes[0].set(title='(a) Reflection', xlabel='Frequency (GHz)', ylabel='S-parameter (dB)', xlim=(24,32), ylim=(-31,0), xticks=np.arange(24,33))
axes[0].legend(loc='lower right', fontsize=10, frameon=False)
axes[0].text(28,-1.1, 'Required: S11, S22 \u2264 \u221210 dB\n26.5\u201329.5 GHz', ha='center', va='top', fontsize=10)
axes[0].text(.48,.13, f'S11 margin: {margins[0]:+.2f} dB\nS22 margin: {margins[1]:+.2f} dB', transform=axes[0].transAxes, fontsize=10)

axes[1].axvspan(25.5,30.5, color=teal, alpha=.10)
for lo,hi in [(24,25),(31,32)]:
    axes[1].axvspan(lo,hi, color=amber, alpha=.09)
    axes[1].hlines(-15,lo,hi,color=amber,ls='--',lw=1.6)
axes[1].hlines(-3,25.5,30.5,color=amber,ls='--',lw=1.6, label='Spec limits')
axes[1].plot(f,s21,'o-',color=teal,ms=4,label='S21')
bad = np.zeros(17,dtype=bool)
bad[3:14] = s21[3:14] < -3
bad[:3] = s21[:3] > -15
bad[14:] = s21[14:] > -15
assert bad.sum() == 6
axes[1].scatter(f[bad],s21[bad],s=65,facecolors='none',edgecolors=amber,linewidths=1.5,zorder=5,label='Unmet sampled point')
axes[1].set(title='(b) Transmission', xlabel='Frequency (GHz)', ylabel='S21 (dB)', xlim=(24,32), ylim=(-31,0), xticks=np.arange(24,33))
axes[1].legend(loc='lower center',fontsize=10,frameon=False,ncol=2)
axes[1].text(28,-7.2, 'Pass band: S21 \u2265 \u22123 dB, 25.5\u201330.5 GHz\nStop bands: S21 \u2264 \u221215 dB, 24\u201325 / 31\u201332 GHz', ha='center', va='top', fontsize=10)
axes[1].text(28,-16, f'Pass-band margin: {margins[2]:+.2f} dB\nStop-band margin: {margins[3]:+.2f} dB', ha='center', va='top',fontsize=10)
for ax in axes:
    ax.grid(alpha=.18)
save_figure(fig, 'candidate-response')

fig, ax = plt.subplots(figsize=(7.2, 4.6), layout='constrained')
hist = data['history']
x = np.arange(1, len(hist)+1)
vals = [r['value'] for r in hist]
ax.step(x, vals, where='post', color=teal, lw=2)
ax.scatter(x, vals, color=teal, s=28, zorder=3)
ax.axhline(0, color=ink, ls='--', lw=1)
ax.text(1,.12, '0 dB margin: all four sampled requirements pass', fontsize=10, va='bottom')
ax.set(xlabel='Best-so-far record number (not HFSS query count)', ylabel='Worst specification margin (dB)', title='Fixed profile: p01 + 0.075 mm bridges', ylim=(-6.4, .75), xticks=[1, 5, 9, 13, 17])
ax.annotate('\u22125.90 dB', (1, vals[0]), xytext=(6, 10), textcoords='offset points')
ax.annotate('\u22122.39 dB', (17, vals[-1]), xytext=(-62, 12), textcoords='offset points')
ax.text(.03, .70, '2026-08-13 to 08-24\n17 records; endpoint change: +3.51 dB', transform=ax.transAxes, fontsize=10)
assert hist[7]['round'] == 73 and hist[7]['value'] == -3.14
ax.annotate('R73 candidate\n(record 8)', (8, vals[7]), xytext=(8,-4.45), ha='center', fontsize=10, arrowprops={'arrowstyle':'-', 'color':ink})
ax.grid(alpha=.18)
save_figure(fig, 'best-so-far')

fig, ax = plt.subplots(figsize=(7.2, 4.8), layout='constrained')
r73 = data['online_r73']
rounds = [r['iter'] for r in r73]
v = [r['wm'] for r in r73]
cum = np.maximum.accumulate([-3.15] + v)[1:]
assert cum[5] == -3.15 and cum[6] == -3.14 and np.all(np.diff(cum) >= 0)
ax.plot(rounds, v, 'o-', color=amber, ms=5, label='Best returned in each batch')
ax.step(rounds, cum, where='post', color=teal, lw=2, label='Best-so-far, including starting record')
ax.set(xlabel='R73 iteration (9 HFSS search results per iteration)', ylabel='Worst specification margin (dB)', title='R73 multi-candidate online feedback', xticks=[1, 3, 6, 9, 12, 15], ylim=(-10.5, -2.3))
ax.grid(alpha=.18)
ax.legend(loc='lower left', fontsize=10, frameon=False)
ax.annotate('Iteration 7: \u22123.15 \u2192 \u22123.14 dB\nRecord gain: +0.01 dB', (7, -3.14), xytext=(1.5, -5.4), arrowprops={'arrowstyle':'-', 'color': ink}, fontsize=10)
ax.text(.98,.97, 'Starting record: \u22123.15 dB', transform=ax.transAxes, ha='right', va='top', fontsize=10)
save_figure(fig, 'online-feedback')
print('Wrote four SVG/PNG figure pairs from frozen data; archived composites retained.')
