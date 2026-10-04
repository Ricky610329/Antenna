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
FIGURES = ('batch-flow', 'candidate-geometry', 'candidate-response', 'best-so-far', 'online-feedback')
for name in FIGURES[1:]:
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
assert [img.get('data-figure') for img in a.images] == list(FIGURES[1:])
for number, (name, figure) in enumerate(zip(FIGURES, a.figures), start=1):
    target = f'fig-{name}'
    caption = f'caption-{name}'
    assert figure.get('aria-labelledby') == caption and caption in a.ids, target
    # Require a linked first explanation immediately before its figure, so splitting
    # the plots cannot silently leave an orphan caption or an obsolete cross-reference.
    intro = re.search(rf'<p class="figure-intro">(?:(?!</p>).)*href="#{target}"(?:(?!</p>).)*</p>\s*<figure id="{target}"', s, re.S)
    assert intro, f'Missing first-use paragraph directly before {target}'
    assert re.search(rf'<figcaption id="{caption}">圖 {number}\s', s), caption
assert '圖 3(c)' not in s and '圖 4(a)' not in s and '圖 4(b)' not in s
for img in a.images:
    assert img.get('alt') and img['src'].startswith('data:image/svg+xml;base64,')
    ET.fromstring(base64.b64decode(img['src'].split(',',1)[1], validate=True))
    assert base64.b64decode(img['src'].split(',',1)[1]) == (BASE / 'figures' / f"{img['data-figure']}.svg").read_bytes()
assert not re.search(r'<(?:script|link)\b[^>]+(?:src|href)="https?://',s)
assert '\ufffd' not in s and '<html lang="zh-Hant">' in s
assert all(f'id="ref{i}"' in s for i in range(1,9))
data = json.loads((BASE / 'paper-evidence-data.json').read_text(encoding='utf-8'))
assert round(data['history'][-1]['value'] - data['history'][0]['value'],2) == 3.51
path.write_text(s, encoding='utf-8')
print(f'PASS: balanced HTML; {len(a.ids)} unique ids; internal/local targets resolve among {len(a.links)} links; 5 numbered figures with first-use explanations; 4 embedded SVGs match their source files; no external assets. External URLs are not fetched by this check.')
print(f'HTML size: {path.stat().st_size:,} bytes. Browser rendering was not performed by this check.')
