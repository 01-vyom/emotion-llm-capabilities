"""Render docs/mats-writeup.md to a self-contained HTML (images inlined) and a PDF.

Run from the repo root:  python docs/build_mats_writeup.py
Needs the `markdown` package; the PDF step uses headless Google Chrome if present.
For the Google Doc: open the HTML in a browser, select all, and paste into the doc
(images and tables carry over), or File > Import the .html.
"""

from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
SRC = DOCS / "mats-writeup.md"
HTML = DOCS / "mats-writeup.html"
PDF = DOCS / "mats-writeup.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--blue:#2a78d6;--surface:#fcfcfb}
*{box-sizing:border-box}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink);
  background:#f4f4f1;line-height:1.55;font-size:15.5px;margin:0;padding:36px 16px}
main{max-width:860px;margin:0 auto;background:var(--surface);padding:52px 60px;
  border:1px solid rgba(11,11,11,.08);border-radius:6px}
h1{font-size:27px;line-height:1.22;margin:0 0 10px}
h2{font-size:20px;margin:36px 0 10px;padding-bottom:5px;border-bottom:2px solid var(--grid)}
h3{font-size:16.5px;margin:24px 0 8px}
p{margin:9px 0} li{margin:4px 0} ul,ol{padding-left:24px}
a{color:var(--blue)}
blockquote{margin:12px 0;padding:8px 16px;background:#f4f7fb;border-left:3px solid var(--blue);color:var(--ink2)}
code{font-family:ui-monospace,Menlo,monospace;font-size:.86em;background:#f0efec;padding:1px 5px;border-radius:4px}
hr{border:none;border-top:1px solid var(--grid);margin:30px 0}
img{max-width:100%;height:auto;display:block;margin:16px auto 6px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:.88em}
th,td{border:1px solid var(--grid);padding:5px 9px;text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}
th{background:#f4f4f1}
@media print{body{background:#fff;padding:0;font-size:11px}main{border:none;padding:0 3mm;max-width:none}
  h2,h3{page-break-after:avoid} img,table,blockquote{page-break-inside:avoid}}
@page{margin:15mm 13mm}
"""


def inline_images(html: str) -> str:
    def repl(m: re.Match) -> str:
        src = m.group(1)
        path = (DOCS / src).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:image/png;base64,{data}"'

    return re.sub(r'src="(figures/[^"]+\.png)"', repl, html)


def main() -> None:
    body = markdown.markdown(SRC.read_text(), extensions=["tables", "sane_lists"])
    title = re.search(r"<h1>(.*?)</h1>", body).group(1)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{re.sub('<[^>]+>', '', title)}</title><style>{CSS}</style></head>"
            f"<body><main>{inline_images(body)}</main></body></html>")
    HTML.write_text(page)
    print("wrote", HTML)
    if Path(CHROME).exists():
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", HTML.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("wrote", PDF)


if __name__ == "__main__":
    main()
