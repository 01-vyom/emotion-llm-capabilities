"""Render docs/mats-writeup.md to a self-contained HTML (images inlined) and a PDF.

Run from the repo root:  python docs/build_mats_writeup.py
Needs the `markdown` package; the PDF step uses headless Google Chrome if present.
Use --soffice /path/to/soffice to print the same HTML through LibreOffice instead.
For the Google Doc: open the HTML in a browser, select all, and paste into the doc
(images and tables carry over), or File > Import the .html.
"""

from __future__ import annotations

import argparse
import base64
import re
import subprocess
import tempfile
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
.figure-image{margin-bottom:4px;page-break-after:avoid}
.figure-caption{margin:4px 0 18px;color:#384047;font-size:.91em;page-break-inside:avoid}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:.88em}
th,td{border:1px solid var(--grid);padding:5px 9px;text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}
th{background:#f4f4f1}
@media print{body{background:#fff;padding:0;font-size:11px}main{border:none;padding:0 3mm;max-width:none}
  h2,h3{page-break-after:avoid} img,table,blockquote{page-break-inside:avoid}}
@page{margin:15mm 13mm}
@media screen and (max-width:620px){body{padding:0}main{padding:24px 16px;border:0}
  h1{font-size:24px}th,td{padding:4px;font-size:13px}}
"""

# Writer's HTML importer uses screen styles and intrinsic image dimensions.
# Give its temporary print copy explicit paper, typography and figure sizes.
WRITER_CSS = """
@page{size:letter;margin:0.65in 0.7in}
body,p,li,td,th{font-family:Arial,sans-serif;color:#171918}
body{background:white}p,li{font-size:10.5pt;line-height:135%}
p{margin:0 0 6pt}li{margin:0 0 3pt}
h1,h2,h3{font-family:Arial,sans-serif;page-break-after:avoid}
h1{font-size:20pt;margin:0 0 9pt}h2{font-size:15pt;margin:13pt 0 5pt}
h3{font-size:12pt;margin:10pt 0 4pt}
table{border-collapse:collapse;width:100%}th,td{font-size:9.5pt;padding:4pt;
border:0.5pt solid #d7dcde;vertical-align:top}th{background:#edf1f2}
blockquote{margin:4pt 12pt}
p.figure-image{page-break-after:avoid;margin-bottom:2pt}
p.figure-caption{page-break-inside:avoid;font-size:9.5pt;color:#384047;margin:2pt 0 9pt}
"""


def print_with_writer(page: str, executable: str) -> None:
    with tempfile.TemporaryDirectory(dir=DOCS) as directory:
        work = Path(directory)
        source = work / HTML.name
        printable = re.sub(r"<style>.*?</style>", f"<style>{WRITER_CSS}</style>", page, flags=re.S)
        printable = printable.replace("<img ", '<img width="666" ')
        source.write_text(printable)
        subprocess.run([executable, f"-env:UserInstallation={(work / 'profile').as_uri()}",
                        "--headless", "--convert-to", "pdf:writer_pdf_Export",
                        "--outdir", str(work), str(source)], check=True, timeout=120)
        (work / PDF.name).replace(PDF)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soffice", help="LibreOffice executable to use for the PDF")
    args = parser.parse_args()
    body = markdown.markdown(SRC.read_text(), extensions=["tables", "sane_lists"])
    body = re.sub(r'<p>(<img [^>]+/>)</p>\s*<p>(?=<strong>Figure )',
                  r'<p class="figure-image">\1</p>\n<p class="figure-caption">', body)
    title = re.search(r"<h1>(.*?)</h1>", body).group(1)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{re.sub('<[^>]+>', '', title)}</title><style>{CSS}</style></head>"
            f"<body><main>{inline_images(body)}</main></body></html>")
    HTML.write_text(page)
    print("wrote", HTML)
    if args.soffice:
        print_with_writer(page, args.soffice)
        print("wrote", PDF)
    elif Path(CHROME).exists():
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", HTML.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("wrote", PDF)


if __name__ == "__main__":
    main()
