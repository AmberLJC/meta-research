import http.server, socketserver, os, markdown, glob, re
from pathlib import Path

PORT = 8421
POSTS_DIR = Path(__file__).parent

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #fafafa; color: #111; line-height: 1.7; }
.nav { background: white; border-bottom: 1px solid #e5e5e5; padding: 16px 0; position: sticky; top: 0; z-index: 10; }
.nav-inner { max-width: 720px; margin: 0 auto; padding: 0 24px;
             display: flex; align-items: center; gap: 16px; }
.nav a { text-decoration: none; color: #111; font-size: 14px; }
.nav .logo { font-weight: 600; font-size: 16px; }
.nav .back { color: #888; font-size: 13px; }
.nav .back:hover { color: #111; }
.index { max-width: 720px; margin: 48px auto; padding: 0 24px; }
.index h1 { font-size: 36px; font-weight: 300; margin-bottom: 8px; letter-spacing: -0.5px; }
.index .sub { color: #666; font-size: 15px; margin-bottom: 40px; }
.post-list { border-top: 1px solid #e5e5e5; }
.post-item { display: flex; align-items: flex-start; gap: 24px; padding: 28px 0; border-bottom: 1px solid #e5e5e5;
             text-decoration: none; color: inherit; transition: background 0.15s; }
.post-item:hover { background: #f5f5f5; margin: 0 -16px; padding-left: 16px; padding-right: 16px; }
.post-num { font-size: 40px; font-family: monospace; color: #ddd; font-weight: 300; width: 56px; flex-shrink: 0; line-height: 1; }
.post-meta { flex: 1; }
.post-tag { font-size: 11px; font-family: monospace; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.post-title { font-size: 20px; font-weight: 500; margin-bottom: 6px; line-height: 1.3; }
.post-desc { font-size: 14px; color: #666; }
.post-arrow { width: 36px; height: 36px; border: 1px solid #e5e5e5; border-radius: 50%;
              display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 4px; color: #999; font-size: 16px; }
.post-item:hover .post-arrow { background: #111; color: white; border-color: #111; }
article { max-width: 720px; margin: 48px auto; padding: 0 24px 80px; }
article h1 { font-size: 36px; font-weight: 500; letter-spacing: -0.5px; margin-bottom: 12px; line-height: 1.2; }
article .tagline { color: #666; font-size: 16px; font-style: italic; margin-bottom: 32px; }
article blockquote { border-left: 3px solid #111; padding-left: 20px; margin: 32px 0; color: #444; font-size: 15px; line-height: 1.6; }
article h2 { font-size: 22px; font-weight: 600; margin-top: 48px; margin-bottom: 16px; }
article p { margin-bottom: 16px; font-size: 16px; }
article pre { background: #111; color: #e8e8e8; padding: 20px; border-radius: 8px; overflow-x: auto;
              font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.6; margin: 24px 0; }
article code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 13px; font-family: monospace; }
article pre code { background: none; padding: 0; }
article table { width: 100%; border-collapse: collapse; margin: 24px 0; font-size: 14px; }
article th { background: #111; color: white; padding: 10px 14px; text-align: left; }
article td { padding: 10px 14px; border-bottom: 1px solid #e5e5e5; }
article tr:hover td { background: #f9f9f9; }
article ul, article ol { padding-left: 24px; margin-bottom: 16px; }
article li { margin-bottom: 6px; }
article strong { font-weight: 600; }
article em { color: #444; }
article hr { border: none; border-top: 1px solid #e5e5e5; margin: 40px 0; }
.footer-note { font-size: 13px; color: #999; font-style: italic; margin-top: 48px; padding-top: 24px; border-top: 1px solid #e5e5e5; }
"""

POSTS = [
    {"slug": "attention-scaling", "file": "attention-scaling.md",
     "title": "Why Transformers Divide Attention by √d",
     "desc": "What happens when dot products grow too large — and how a single constant prevents softmax from becoming useless.",
     "tag": "architecture"},
    {"slug": "layer-normalization", "file": "layer-normalization.md",
     "title": "Why Training Collapses Without Layer Normalization",
     "desc": "What happens when activations drift — and how LayerNorm keeps every layer speaking the same language.",
     "tag": "optimization"},
    {"slug": "learning-rate-warmup", "file": "learning-rate-warmup.md",
     "title": "Why Cold-Starting with a High Learning Rate Kills Training",
     "desc": "What happens in the first 1000 steps — and why warmup is non-negotiable.",
     "tag": "optimization"},
]

def page(title, body, back=False):
    nav_extra = '<a class="back" href="/">← All posts</a>' if back else ''
    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} | Orchestra</title>
<style>{CSS}</style></head><body>
<nav class="nav"><div class="nav-inner">
  <a class="logo" href="/">Orchestra</a>
  <span style="color:#ddd">|</span>
  <span style="color:#888;font-size:13px">Intro to AI Research</span>
  {nav_extra}
</div></nav>
{body}
</body></html>"""

def index_html():
    items = ""
    for i, p in enumerate(POSTS, 1):
        items += f"""<a class="post-item" href="/{p['slug']}">
  <span class="post-num">0{i}</span>
  <div class="post-meta">
    <div class="post-tag"># {p['tag']}</div>
    <div class="post-title">{p['title']}</div>
    <div class="post-desc">{p['desc']}</div>
  </div>
  <div class="post-arrow">→</div>
</a>"""
    body = f"""<div class="index">
  <p style="font-size:11px;font-family:monospace;color:#999;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px">AI RESEARCH FUNDAMENTALS</p>
  <h1>Learn AI Research,<br><span style="font-weight:500">One Concept</span> at a Time.</h1>
  <p class="sub">Bite-sized lessons, toy models, and hands-on experiments.</p>
  <div class="post-list">{items}</div>
</div>"""
    return page("Intro to AI Research", body)

def post_html(p):
    md_text = (POSTS_DIR / p['file']).read_text()
    # strip h1 and tagline from markdown (we'll render them separately)
    lines = md_text.split('\n')
    h1 = lines[0].lstrip('# ').strip()
    tagline = ''
    rest_start = 1
    for i, l in enumerate(lines[1:], 1):
        stripped = l.strip()
        if stripped.startswith('*') and stripped.endswith('*'):
            tagline = stripped.strip('*')
            rest_start = i + 1
            break
        elif stripped:
            rest_start = i
            break
    rest_md = '\n'.join(lines[rest_start:])
    html_body = markdown.markdown(rest_md, extensions=['tables', 'fenced_code'])
    # fix footer italic
    html_body = html_body.replace('<p><em>Experiments', '<p class="footer-note"><em>Experiments')
    body = f"""<article>
  <h1>{h1}</h1>
  <p class="tagline">{tagline}</p>
  {html_body}
</article>"""
    return page(h1, body, back=True)

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        path = self.path.strip('/')
        if path == '' or path == 'index':
            html = index_html()
        else:
            post = next((p for p in POSTS if p['slug'] == path), None)
            if post:
                html = post_html(post)
            else:
                self.send_response(404); self.end_headers()
                self.wfile.write(b'Not found'); return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

print(f"Serving at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
