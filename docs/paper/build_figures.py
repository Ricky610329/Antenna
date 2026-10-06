"""Rebuild paper figures from the adjacent frozen evidence data; never runs HFSS."""
import json
import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
    """Save a named manuscript or audit asset from the frozen evidence."""
    svg = out / f'{basename}.svg'
    fig.savefig(svg, metadata={'Date': None})
    fig.savefig(out / f'{basename}.png', dpi=165)
    plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')


def paper_coordinates(points):
    """HFSS (X=row, Y=col) -> paper (u=Y, v=5-X), as imshow(origin='upper').

    Apply the proper clockwise rotation + translation to every polygon vertex.
    Merely transposing row/column or moving the lower-left anchor mirrors or
    offsets the 0.21 mm overlapping squares. The input bits stay untouched.
    """
    points = np.asarray(points, dtype=float)
    return np.column_stack((points[:, 1], 5 - points[:, 0]))


# Orientation and extent checks independent of a candidate's visual symmetry.
assert np.allclose(paper_coordinates([(0, 0), (5, 0), (0, 5), (5, 5)]),
                   [(0, 5), (0, 0), (5, 5), (5, 0)])
assert np.linalg.det(np.asarray([[0., 1.], [-1., 0.]])) == 1  # no mirror
assert np.allclose(paper_coordinates([(4.8, 0), (5.01, .21)]), [(0, .2), (.21, -.01)])
assert np.allclose(paper_coordinates([(5, 2.5), (0, 2.5)]), [(2.5, 0), (2.5, 5)])
# All 625 nominal cell centers must occupy the same visual row/column as an
# independent imshow(matrix, origin='upper', extent=(0,5,0,5)) reference.
ii, jj = np.indices((25, 25))
center_display = paper_coordinates(np.column_stack(((ii.ravel()+.5)*.2, (jj.ravel()+.5)*.2)))
assert np.allclose(center_display[:, 0], (jj.ravel()+.5)*.2)
assert np.allclose(center_display[:, 1], (24.5-ii.ravel())*.2)


def geometry_figure(matrix, bridge_side, dual, basename, title):
    bridge_sites, omitted = namespace['diag_bridge_sites'](matrix, bridge_side, .2)
    assert omitted == 0
    fig, ax = plt.subplots(figsize=(6.4, 6.6), layout='constrained')
    for row, col in zip(*np.nonzero(matrix)):
        x, y, side = row * .2, col * .2, .21
        corners = [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]
        transformed = paper_coordinates(corners)
        assert np.allclose(transformed.min(axis=0), [y, 5-x-side])
        ax.add_patch(Polygon(transformed, closed=True, facecolor=ink, edgecolor='none'))
    for sx, sy, width in bridge_sites:
        delta = width / np.sqrt(2)
        corners = [(sx-delta,sy),(sx,sy+delta),(sx+delta,sy),(sx,sy-delta)]
        ax.add_patch(Polygon(paper_coordinates(corners), closed=True, facecolor=amber, edgecolor='none'))
    ax.set(xlim=(-.18,5.18), ylim=(-.40,5.40), xlabel='HFSS Y (mm)',
           ylabel='5 mm \u2212 HFSS X (mm)', title=title)
    ax.set_aspect('equal')
    # Port locations follow simulator assignments: single P1 X=27.5;
    # dual P1 X=12.5, P2 X=-7.5. Mark only their pixel-region connections.
    # Keep transparent port labels outside the metal region: an opaque label
    # patch can otherwise look like an etched opening in the pixel geometry.
    feeds = [(5, '^', 'P1', -.22)]
    if dual:
        feeds.append((0, 'v', 'P2', 5.22))
    for physical_x, marker, label, text_v in feeds:
        u, v = paper_coordinates([(physical_x, 2.5)])[0]
        ax.scatter([u],[v],marker=marker,s=90,c=teal,edgecolor='none',zorder=10)
        ax.text(u+.22,text_v,label,fontsize=11,color=teal,va='center')
    fig.supxlabel(f'{len(bridge_sites)} orange bridges: square side {bridge_side:g} mm\n'
                  '0.01 mm pixel overlap; P1 feed connection at bottom\n'
                  'External feeds / substrate / ground omitted', fontsize=10)
    save_figure(fig, basename)
    return len(bridge_sites)


assert geometry_figure(bits, .075, True, 'candidate-geometry',
                       'Dual-port filter: pixel-region geometry') == 67

# Audit-only comparison: keep the former error visible beside the corrected
# polygons and an independent raster convention, using this asymmetric design.
fig, audit_axes = plt.subplots(1, 3, figsize=(11.4,4.35), layout='constrained')
for index, ax in enumerate(audit_axes[:2]):
    transform = paper_coordinates if index else lambda p: np.asarray(p)
    for row, col in zip(*np.nonzero(bits)):
        x, y, side = row*.2, col*.2, .21
        ax.add_patch(Polygon(transform([(x,y),(x+side,y),(x+side,y+side),(x,y+side)]),facecolor=ink,edgecolor='none'))
    for sx, sy, width in sites:
        delta = width/np.sqrt(2)
        ax.add_patch(Polygon(transform([(sx-delta,sy),(sx,sy+delta),(sx+delta,sy),(sx,sy-delta)]),facecolor=amber,edgecolor='none'))
    ax.set(xlim=(-.15,5.15),ylim=(-.15,5.15))
    if index:
        for y, marker, label in [(0,'^','P1'),(5,'v','P2')]:
            ax.scatter([2.5],[y],marker=marker,s=70,color=teal,edgecolor='white',zorder=10)
            ax.text(2.75, .25 if y == 0 else 4.65,label,fontsize=10,bbox={'facecolor':'white','edgecolor':'none','pad':1})
    else:
        for x, marker, label in [(5,'<','P1'),(0,'>','P2')]:
            ax.scatter([x],[2.5],marker=marker,s=70,color=teal,edgecolor='white',zorder=10)
            ax.text(4.4 if x == 5 else .15,2.8,label,fontsize=10,bbox={'facecolor':'white','edgecolor':'none','pad':1})
audit_axes[2].imshow(bits,origin='upper',extent=(0,5,0,5),interpolation='nearest',cmap=matplotlib.colors.ListedColormap(['white',ink]),vmin=0,vmax=1)
for y, marker, label in [(0,'^','P1'),(5,'v','P2')]:
    audit_axes[2].scatter([2.5],[y],marker=marker,s=70,color=teal,edgecolor='white',zorder=10)
    audit_axes[2].text(2.75,.25 if y == 0 else 4.65,label,fontsize=10,bbox={'facecolor':'white','edgecolor':'none','pad':1})
for ax, title, xlabel, ylabel in zip(audit_axes,
        ['(a) Former orientation: incorrect','(b) Corrected physical polygons','(c) Independent matrix reference'],
        ['HFSS X','HFSS Y','Matrix column j (scaled to mm)'],
        ['HFSS Y','5 mm - HFSS X','Display v (mm); row i increases downward']):
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel,aspect='equal')
    ax.tick_params(labelsize=9)
fig.supxlabel('Same frozen 625 bits throughout. (b) uses actual 0.21 mm pixels and 67 bridges; (c) checks orientation only.',fontsize=10)
save_figure(fig,'geometry-orientation-audit')

# Single-port figures use one formal handoff evaluation throughout; the earlier
# R54 score belongs to a different evaluation and is never paired with these curves.
single = json.loads((BASE / 'single-port-evidence.json').read_text(encoding='utf-8'))
single_bits = np.asarray(single['candidate']['bits'])
assert single_bits.shape == (25, 25) and np.isin(single_bits, [0, 1]).all()
assert geometry_figure(single_bits, .1, False, 'single-geometry',
                       'Single-port antenna: pixel-region geometry') == 12
sf = np.asarray(single['frequency_ghz'])
ss11 = np.asarray(single['response_db']['S11'])
sgain = np.asarray(single['response_db']['RealizedGainTotal'])
sm1 = float(-10-ss11[5:12].max())
sm2 = float(sgain[5:12].min()-4)
assert [round(sm1, 2), round(sm2, 2), round(min(sm1, sm2), 2)] == [1.13, .77, .77]
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), layout='constrained')
for ax in axes:
    ax.axvspan(26.5, 29.5, color=teal, alpha=.10)
    ax.set(xlim=(24,32), xticks=np.arange(24,33), xlabel='Frequency (GHz)')
    ax.grid(alpha=.18)
axes[0].plot(sf,ss11,'o-',color=teal,ms=4,label='S11')
axes[0].hlines(-10,26.5,29.5,color=amber,ls='--',lw=1.6)
axes[0].set(title='(a) Reflection',ylabel='S11 (dB)',ylim=(-25,0))
axes[0].text(28,-2,'Required: S11 \u2264 \u221210 dB\n26.5\u201329.5 GHz',ha='center',va='top',fontsize=10)
axes[0].text(30.8,-22, f'S11 margin: {sm1:+.2f} dB',ha='right',fontsize=10)
axes[1].plot(sf,sgain,'o-',color=teal,ms=4)
axes[1].hlines(4,26.5,29.5,color=amber,ls='--',lw=1.6)
axes[1].set(title='(b) Boresight realized gain (theta = phi = 0\N{DEGREE SIGN})',ylabel='Realized gain (dBi)',ylim=(0,8))
axes[1].text(28,7.4,'Required: realized gain \u2265 4 dBi\n26.5\u201329.5 GHz',ha='center',va='top',fontsize=10)
axes[1].text(28,1.1,f'Gain margin: {sm2:+.2f} dB; worst margin: {min(sm1,sm2):+.2f} dB',ha='center',fontsize=10)
save_figure(fig, 'single-response')

rad = single['radiation']
theta = np.asarray(rad['theta_deg'])
theta_rad = np.deg2rad(theta)
window = np.abs(theta) <= 45
zero = int(np.argmin(np.abs(theta)))
fig, ax = plt.subplots(figsize=(7.0, 6.0), subplot_kw={'projection':'polar'}, layout='constrained')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_thetagrids(np.arange(-150,181,30))
ax.set(ylim=(-30,3), yticks=[-30,-20,-10,-3,0], title='28 GHz radiation: normalized total gain')
ax.set_rlabel_position(140)
ax.fill_between(np.deg2rad(np.linspace(-45,45,91)), -30, 3, color=teal, alpha=.08)
for key, color in [('phi0',teal),('phi90',ink)]:
    values = np.asarray(rad[key+'_db'])
    normalized = values-values[zero]
    margin = float(normalized[window].min()+3)
    assert round(margin,2) == round(rad['margin_db'][key],2)
    assert normalized.min() >= -30 and normalized.max() <= 3  # no clipped data
    ax.plot(theta_rad,normalized,color=color,lw=1.7,label=f'HFSS {key.replace("phi", "phi = ")}\N{DEGREE SIGN}')
ax.plot(np.deg2rad(np.linspace(-45,45,91)),np.full(91,-3),color=amber,ls='--',lw=1.5,label='\u22123 dB floor in window')
for angle in [-45,45]:
    ax.plot([np.deg2rad(angle)]*2,[-30,3],color=amber,ls=':',lw=1.2)
ax.set_thetalim(-np.pi,np.pi)  # signed tick locations must not crop the polar domain
assert np.isclose(ax.get_thetamax()-ax.get_thetamin(), 360)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.04),ncol=2,fontsize=10,frameon=False)
fig.supxlabel('Each cut: GainTotal(theta) \u2212 GainTotal(0\N{DEGREE SIGN}), in dB\n'
              'Window margins: phi=0\N{DEGREE SIGN} +0.63 dB; phi=90\N{DEGREE SIGN} +0.50 dB\n'
              'Saved 2\N{DEGREE SIGN} grid: evaluated window endpoints are \u00b144\N{DEGREE SIGN}',fontsize=10)
save_figure(fig, 'single-radiation')

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
print('Wrote seven manuscript SVG/PNG pairs plus one orientation-audit pair.')
