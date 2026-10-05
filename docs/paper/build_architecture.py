"""Draw the reader-facing mechanism; does not run strategies, training, or HFSS.

The diagram is a synthesis of the research workflow and emforge's separation of
strategy-owned state from shared evaluation. It is not a deployment inventory.
See the manuscript's author appendix for implementation and validation scopes.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parent
OUT = BASE / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'antenna-paper-architecture-v3'})
ink, teal, muted = '#213447', '#087d80', '#526876'
fig, ax = plt.subplots(figsize=(12, 8.8))
fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.01)
ax.set(xlim=(0, 12), ylim=(0, 8.8))
ax.axis('off')


def box(x, y, w, h, title, lines=(), color='#edf5f3', dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.035,rounding_size=0.09',
                              facecolor=color, edgecolor=teal if not dashed else muted,
                              linewidth=1.25, linestyle='--' if dashed else '-'))
    ax.text(x+.16, y+h-.22, title, va='top', color=ink, fontsize=12, weight='bold')
    for index, line in enumerate(lines):
        ax.text(x+.16, y+h-.59-index*.28, line, va='top', color=ink, fontsize=11)


def arrow(points, color=teal, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.6,
                linestyle='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                mutation_scale=12, linewidth=1.6, color=color,
                                linestyle='--' if dashed else '-', shrinkA=0, shrinkB=0))


box(.55, 7.63, 5.15, .95, 'Researcher / optional AI assistant',
    ['Define variants, inspect evidence, configure strategies'], '#f3f3f0', True)
box(6.55, 7.63, 4.9, .95, 'Task definition',
    ['Geometry + solver profile; response + target specification'], '#f3f3f0')
arrow([(5.5, 7.6), (5.5, 7.06)], muted, True)
arrow([(11.15, 7.6), (11.15, 6.93)], muted)
ax.text(.65, 7.13, 'STRATEGY-OWNED SEARCH', color=teal, fontsize=12, weight='bold')
ax.text(6.67, 7.13, 'SHARED EVALUATION', color=teal, fontsize=12, weight='bold')
ax.add_patch(FancyBboxPatch((.49, 1.57), 5.57, 5.45,
                          boxstyle='round,pad=0.025,rounding_size=0.1',
                          facecolor='none', edgecolor='#c6d9d6', linewidth=.9))

box(.65, 4.96, 5.25, 1.93, 'Structured search with forward surrogate',
    ['Seeds / structural variation / candidate pool',
     'SM-based selection + score-independent exploration',
     'Own parents, model and update rule'])
box(.65, 3.43, 5.25, 1.25, 'Other stateful search strategies',
    ['Local search / component edits / random reference',
     'Own state and feedback schedule; SM not required'])
box(.65, 1.71, 5.25, 1.4, 'On receiving usable results',
    ['Update search state; accumulate new labels',
     'Refit SM only when this strategy requests it',
     'Then propose one or multiple candidates'])

# The common submission bus merges requests, not strategy clocks.
ax.plot([6.12, 6.12], [4.05, 6.13], color=teal, lw=1.6)
arrow([(5.94, 6.13), (6.12, 6.13), (6.48, 6.13)])
arrow([(5.94, 4.05), (6.12, 4.05), (6.12, 6.13)])
box(6.52, 5.47, 4.88, 1.42, 'Validate, share and schedule',
    ['Identity / deduplication / existing-result reuse',
     'Priority + applicable budgets + in-flight limits',
     'Dispatch new work; keep repeat requests distinct'])
box(6.52, 3.65, 4.88, 1.3, 'HFSS workers',
    ['Build and solve the specified geometry',
     'Return response or failure; record solve cost'])
arrow([(8.96, 5.43), (8.96, 4.99)])
box(6.52, 1.71, 4.88, 1.43, 'Collect, measure and score',
    ['Preserve raw responses and evaluation identity',
     'Apply the task specification; store results',
     'Return results to the requesting strategy / run'])
arrow([(8.96, 3.61), (8.96, 3.18)])
arrow([(6.48, 2.42), (5.94, 2.42)])

# State feedback is local to each strategy; no global training barrier.
arrow([(.61, 2.42), (.25, 2.42), (.25, 5.94), (.61, 5.94)])
arrow([(.25, 4.05), (.61, 4.05)])
ax.text(.65, 1.35, 'Results are shared; update schedules remain strategy-specific.',
        color=teal, fontsize=12, weight='bold')
ax.text(.65, .96, 'Candidate batches, worker jobs and model-training minibatches are different quantities.',
        color=muted, fontsize=11)
ax.text(.65, .59, 'Final design claims use HFSS responses and selected confirmation runs, not surrogate scores.',
        color=muted, fontsize=11)
ax.text(.65, .22, 'Schematic of responsibilities and data flow; each strategy decides when to consume feedback.',
        color=muted, fontsize=10)

svg = OUT / 'architecture.svg'
fig.savefig(svg, metadata={'Date': None})
fig.savefig(OUT / 'architecture.png', dpi=165)
plt.close(fig)
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
               encoding='utf-8')
print('Wrote architecture.svg and architecture.png (logical mechanism only).')
