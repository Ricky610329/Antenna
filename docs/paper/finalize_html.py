"""Embed rebuilt figures and validate the standalone HTML draft, without a browser."""
import base64
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
path = BASE / 'manuscript-draft-2026-10.html'
s = path.read_text(encoding='utf-8')
FIGURES = ('architecture', 'emforge-architecture',
           'single-geometry', 'single-response', 'single-radiation',
           'candidate-geometry', 'candidate-response')
for name in FIGURES:
    figure = (BASE / 'figures' / f'{name}.svg').read_bytes()
    ET.fromstring(figure)
    uri = 'data:image/svg+xml;base64,' + base64.b64encode(figure).decode('ascii')
    pattern = rf'(<img data-figure="{name}" src=")[^"]+(" alt=)'
    s, count = re.subn(pattern, lambda m: m[1]+uri+m[2], s)
    assert count == 1, (name, count)

class Audit(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors, self.links, self.ids, self.images, self.figures = [], [], [], [], [], []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag == 'a': self.links.append(a.get('href',''))
        if tag == 'img': self.images.append(a)
        if tag == 'figure': self.figures.append(a)
        if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:
            self.stack.append(tag)
    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag: self.errors.append((tag,self.stack[-3:]))
        else: self.stack.pop()

a = Audit()
a.feed(s)
a.close()
assert not a.stack and not a.errors, (a.stack,a.errors)
assert len(a.ids) == len(set(a.ids))
for href in a.links:
    u = urlsplit(href)
    if href.startswith('#'): assert href[1:] in a.ids, href
    elif not u.scheme: assert (BASE / unquote(u.path)).exists(), href
assert [figure.get('id') for figure in a.figures] == [f'fig-{name}' for name in FIGURES]
assert [img.get('data-figure') for img in a.images] == list(FIGURES)
for number, (name, figure) in enumerate(zip(FIGURES, a.figures), start=1):
    target = f'fig-{name}'
    caption = f'caption-{name}'
    assert figure.get('aria-labelledby') == caption and caption in a.ids, target
    # Require a linked first explanation immediately before its figure, so splitting
    # the plots cannot silently leave an orphan caption or an obsolete cross-reference.
    intro = re.search(rf'<p class="figure-intro">(?:(?!</p>).)*href="#{target}"(?:(?!</p>).)*</p>\s*<figure id="{target}"', s, re.S)
    assert intro, f'Missing first-use paragraph directly before {target}'
    assert re.search(rf'<figcaption id="{caption}">圖 {number}\s', s), caption
assert 'data-figure="execution-time"' not in s, 'Retired timing figure must not return'
assert '@@' not in s, 'Unresolved manuscript placeholder'
assert '@media(max-width:780px)' not in s, 'Mobile flow rules must not affect print'
for img in a.images:
    assert img.get('alt') and img['src'].startswith('data:image/svg+xml;base64,')
    ET.fromstring(base64.b64decode(img['src'].split(',',1)[1], validate=True))
    assert base64.b64decode(img['src'].split(',',1)[1]) == (BASE / 'figures' / f"{img['data-figure']}.svg").read_bytes()
assert not re.search(r'<(?:script|link)\b[^>]+(?:src|href)="https?://',s)
assert '\ufffd' not in s and '<html lang="zh-Hant">' in s
assert all(f'id="ref{i}"' in s for i in range(1,10))
# Audit the entire figure-reference map, including panel references in prose.
figure_numbers = {name: number for number, name in enumerate(FIGURES, start=1)}
for target, number in re.findall(r'href="#fig-([a-z-]+)"[^>]*>圖\s*(\d+)', s):
    assert int(number) == figure_numbers[target], (target, number)
headings = re.findall(r'<section id="[^"]+"><h2>([IVX]+)\.', s)
assert headings == ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], headings
equations = re.findall(r'<span class="eqno">\((\d+)\)</span>', s)
assert equations == ['1', '2', '3', '4', '5'], equations
table_numbers = re.findall(r'<caption>表 ([IVX]+)\s', s)
assert table_numbers == ['I', 'II', 'III'], table_numbers
data = json.loads((BASE / 'paper-evidence-data.json').read_text(encoding='utf-8'))
assert round(data['history'][-1]['value'] - data['history'][0]['value'],2) == 3.51
single = json.loads((BASE / 'single-port-evidence.json').read_text(encoding='utf-8'))
band = single['frequency_response_definition']['spec_sample_indices']
response = single['response_db']
m_s = -10 - max(response['S11'][i] for i in band)
m_g = min(response['RealizedGainTotal'][i] for i in band) - 4
assert [round(m_s, 2), round(m_g, 2), round(min(m_s, m_g), 2)] == [1.13, 0.77, 0.77]
rad = single['radiation']
theta = rad['theta_deg']
zero = min(range(len(theta)), key=lambda i: abs(theta[i]))
window = [i for i, angle in enumerate(theta) if abs(angle) <= 45]
rad_margins = [min(rad[cut][i] for i in window) - rad[cut][zero] + 3 for cut in ('phi0_db', 'phi90_db')]
assert [round(value, 2) for value in rad_margins] == [0.63, 0.50]
assert max(abs(theta[i]) for i in window) == 44
dual = json.loads((BASE / 'final-candidate-response.json').read_text(encoding='utf-8'))
f = dual['frequency_ghz']
dr = dual['response_db']
reflection = [i for i, x in enumerate(f) if 26.5 <= x <= 29.5]
passband = [i for i, x in enumerate(f) if 25.5 <= x <= 30.5]
stopband = [i for i, x in enumerate(f) if x in {24, 24.5, 25, 31, 31.5, 32}]
dual_margins = [
    -10 - max(dr['S11'][i] for i in reflection),
    -10 - max(dr['S22'][i] for i in reflection),
    min(dr['S21'][i] for i in passband) + 3,
    -15 - max(dr['S21'][i] for i in stopband),
]
assert [round(value, 2) for value in dual_margins] == [1.97, 2.31, -2.39, -2.17]
failed = sum(dr['S11'][i] > -10 or dr['S22'][i] > -10 for i in reflection)
failed += sum(dr['S21'][i] < -3 for i in passband)
failed += sum(dr['S21'][i] > -15 for i in stopband)
assert failed == 6
ideal = 3 * 604800 / 160
assert ideal == 11340
assert round(ideal * 0.50 * 0.95 * 0.90) == 4848
assert round(ideal * 0.70 * 0.95 * 0.90) == 6787
assert 3 * 90 == 270
path.write_text(s, encoding='utf-8')
print(f'PASS: balanced HTML; {len(a.ids)} unique ids; internal/local targets resolve among {len(a.links)} links; {len(FIGURES)} numbered figures with first-use explanations; {len(a.images)} embedded SVGs match their source files; antenna and filter margins, failed filter points, and capacity examples independently recomputed; no external assets. External URLs are not fetched by this check.')
print(f'HTML size: {path.stat().st_size:,} bytes. Browser rendering was not performed by this check.')
