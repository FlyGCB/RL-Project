"""Convert report.md -> report.pdf via Markdown -> HTML -> xhtml2pdf.
Image paths in the markdown (report_assets/*.png) are resolved relative to
the project root and inlined as absolute file:// paths so xhtml2pdf can find them.
"""
import base64
import re
from pathlib import Path
import markdown
from xhtml2pdf import pisa

PROJECT_ROOT = Path(__file__).parent.parent
md_path = PROJECT_ROOT / "report.md"
pdf_path = PROJECT_ROOT / "report.pdf"

md_text = md_path.read_text(encoding="utf-8")
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

# Inline relative images (report_assets/*.png) as base64 data URIs -- xhtml2pdf's
# file:// resolution is unreliable on Windows paths containing spaces.
def inline_image(match):
    rel_path = match.group(1)
    img_path = PROJECT_ROOT / rel_path
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f'src="data:image/png;base64,{b64}"'

html_body = re.sub(r'src="(report_assets/[^"]+)"', inline_image, html_body)

CSS = """
<style>
  @page { size: A4; margin: 2cm; }
  body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
  h1 { font-size: 20pt; margin-top: 0; }
  h2 { font-size: 15pt; margin-top: 22px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
  h3 { font-size: 12.5pt; margin-top: 16px; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
  th, td { border: 1px solid #999; padding: 4px 7px; text-align: left; }
  th { background-color: #eef2f7; }
  img { max-width: 100%; margin: 8px 0; }
  code { background: #f2f2f2; padding: 1px 4px; font-size: 9pt; }
  blockquote { border-left: 3px solid #999; margin: 8px 0; padding-left: 10px; color: #444; }
  strong { color: #111; }
</style>
"""

full_html = f"<html><head><meta charset='utf-8'>{CSS}</head><body>{html_body}</body></html>"

with open(pdf_path, "wb") as f:
    result = pisa.CreatePDF(full_html, dest=f)

if result.err:
    raise SystemExit(f"PDF generation had {result.err} error(s)")

print(f"Wrote {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
