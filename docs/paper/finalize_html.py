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
for name in ('candidate', 'search-history'):
    figure = (BASE / 'figures' / f'{name}.svg').read_bytes()
    ET.fromstring(figure)
    uri = 'data:image/svg+xml;base64,' + base64.b64encode(figure).decode('ascii')
    pattern = rf'(<img data-figure="{name}" src=")[^"]+(" alt=)'
    s, count = re.subn(pattern, lambda m: m[1]+uri+m[2], s)
    assert count == 1, (name, count)
path.write_text(s, encoding='utf-8')

class Audit(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors, self.links, self.ids, self.images = [], [], [], [], []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag == 'a': self.links.append(a.get('href',''))
        if tag == 'img': self.images.append(a)
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
assert len(a.images) == 2
for img in a.images:
    assert img.get('alt') and img['src'].startswith('data:image/svg+xml;base64,')
    ET.fromstring(base64.b64decode(img['src'].split(',',1)[1], validate=True))
assert not re.search(r'<(?:script|link)\b[^>]+(?:src|href)="https?://',s)
assert '\ufffd' not in s and '<html lang="zh-Hant">' in s
assert all(f'id="ref{i}"' in s for i in range(1,9))
data = json.loads((BASE / 'paper-evidence-data.json').read_text(encoding='utf-8'))
assert round(data['history'][-1]['value'] - data['history'][0]['value'],2) == 3.51
print(f'PASS: balanced HTML; {len(a.ids)} unique ids; internal/local targets resolve among {len(a.links)} links; 2 embedded SVGs; no external assets. External URLs are not fetched by this check.')
print(f'HTML size: {path.stat().st_size:,} bytes. Browser rendering was not performed by this check.')
