"""
Build the docs/ folder that GitHub Pages serves.

Everything in docs/ is derived from something else in the repo, so this script
is the single place that derivation happens. Run it after regenerating figures
or editing the report.

  reports/IS7085_Governance_Report.pdf  ->  docs/report.pdf
  reports/report.html                   ->  docs/report.html   (paths + screen CSS)
  reports/MODEL_CARD.md                 ->  docs/MODEL_CARD.md
                                        ->  docs/model-card.html  (rendered)
  figures/*.png                         ->  docs/figures/

The notebook is converted separately, because that needs jupyter:

  python -m nbconvert --to html --output-dir docs --output notebook.html \
      notebooks/resume_screening_governance.ipynb
  python build_site.py            # re-adds the notebook's back link

Run:  python build_site.py
"""

import os
import re
import shutil

import mistune

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOCS = os.path.join(BASE, "docs")
REPORTS = os.path.join(BASE, "reports")
FIGURES = os.path.join(BASE, "figures")

REPORT_PDF = "IS7085_Governance_Report.pdf"

# The report stylesheet targets Letter print at 9.15pt justified serif. That is
# close to unreadable full-width in a browser, so the web edition gets a screen
# layer on top. The print rules are untouched, so printing from the browser
# still produces the original layout.
SCREEN_CSS = """
/* --- screen-only overrides (web edition; print rules above are untouched) --- */
@media screen {
  body { max-width: 46rem; margin: 0 auto; padding: 2.2rem 1.5rem 3rem;
         font-size: 11.5pt; line-height: 1.55; background: #fdfdfb; }
  h1 { font-size: 26pt; }
  h2 { font-size: 15pt; margin-top: 26px; }
  h3 { font-size: 12.2pt; margin-top: 18px; }
  table { font-size: 9.6pt; }
  figure img { width: 100%; max-width: 100%; }
  figcaption { font-size: 9.4pt; }
  .meta { font-size: 9.4pt; flex-wrap: wrap; }
  .sub { font-size: 10.6pt; }
  code { font-size: 10.2pt; }
  .foot { font-size: 8.8pt; }
  .pb { break-before: auto; }
  .backlink { display: block; font-family: "DejaVu Sans", Helvetica, sans-serif;
              font-size: 9.6pt; margin-bottom: 18px; color: #2a78d6;
              text-decoration: none; }
  .backlink:hover { text-decoration: underline; }
}
@media print { .backlink { display: none; } }
@media screen and (max-width: 640px) {
  body { font-size: 11pt; padding: 1.4rem 1rem 2.5rem; }
  h1 { font-size: 21pt; }
  table { font-size: 8.6pt; }
  p, li { text-align: left; }
}
"""

MODEL_CARD_PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Card — Resume Screening Shortlist Ranker</title>
<meta name="description" content="Model card for the governed resume screening ranker: what it is for, how it performs, and what is still broken.">
<style>
:root {{
  --ink: #16150f; --muted: #52514e; --rule: #d8d7d2; --hair: #e6e4de;
  --blue: #2a78d6; --red: #e34948; --wash: #f6f5f1; --page: #fdfdfb;
}}
* {{ box-sizing: border-box; }}
body {{ font-family: "DejaVu Serif", Georgia, serif; font-size: 11.5pt; line-height: 1.55;
        color: var(--ink); background: var(--page);
        max-width: 46rem; margin: 0 auto; padding: 2.2rem 1.5rem 3rem; }}
h1, h2, h3 {{ font-family: "DejaVu Sans", Helvetica, Arial, sans-serif; }}
h1 {{ font-size: 26pt; line-height: 1.15; margin: 0 0 6px; letter-spacing: -0.4px; }}
h2 {{ font-size: 15pt; margin: 26px 0 6px; padding-bottom: 4px;
      border-bottom: 1.6px solid var(--ink); }}
h3 {{ font-size: 12.2pt; margin: 18px 0 4px; }}
p {{ margin: 0 0 9px; }}
a {{ color: var(--blue); }}
.sub {{ font-family: "DejaVu Sans", Helvetica, sans-serif; color: var(--muted);
        font-size: 10.6pt; margin: 0 0 14px; }}
table {{ width: 100%; border-collapse: collapse; display: block; overflow-x: auto;
         font-family: "DejaVu Sans", Helvetica, sans-serif; font-size: 9.6pt;
         margin: 10px 0 14px; }}
th {{ text-align: left; border-bottom: 1.4px solid var(--ink); padding: 6px 7px; font-weight: 600; }}
td {{ border-bottom: 0.7px solid var(--hair); padding: 5px 7px; vertical-align: top; }}
ol, ul {{ margin: 8px 0 12px; padding-left: 20px; }}
li {{ margin-bottom: 6px; }}
code {{ font-family: "DejaVu Sans Mono", ui-monospace, monospace; font-size: 10.2pt;
        background: #f2f1ec; padding: 1px 4px; border-radius: 2px; }}
strong {{ font-weight: 700; }}
.backlink {{ display: inline-block; font-family: "DejaVu Sans", Helvetica, sans-serif;
             font-size: 9.6pt; margin-bottom: 18px; color: var(--blue);
             text-decoration: none; }}
.backlink:hover {{ text-decoration: underline; }}
.foot {{ font-family: "DejaVu Sans", Helvetica, sans-serif; font-size: 8.8pt;
         color: #6b6a63; border-top: 1px solid var(--rule); padding-top: 10px;
         margin-top: 30px; }}
@media (max-width: 640px) {{
  body {{ padding: 1.4rem 1rem 2.5rem; }}
  h1 {{ font-size: 21pt; }}
}}
</style></head><body>

<a class="backlink" href="./">&larr; Project home</a>

<h1>{title}</h1>
<p class="sub">The governed shortlist ranker for software engineering applicants</p>

{body}

<div class="foot">Source: <code>reports/MODEL_CARD.md</code>. Every number here is generated by
<code>src/</code>, seeded at 42, and checked by <code>tests/verify_report.py</code>.</div>

</body></html>
"""

NB_BANNER = """<div id="nb-backlink" style="font-family:'DejaVu Sans',Helvetica,Arial,sans-serif;
 font-size:13px;padding:10px 18px;border-bottom:1px solid #d8d7d2;background:#f6f5f1;">
<a href="./" style="color:#2a78d6;text-decoration:none;">&larr; Project home</a>
<span style="color:#52514e;"> &nbsp;&middot;&nbsp; Notebook &mdash; Responsible AI Resume Screening</span>
</div>
"""


def copy_figures():
    dest = os.path.join(DOCS, "figures")
    os.makedirs(dest, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(FIGURES)):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(FIGURES, f), os.path.join(dest, f))
            n += 1
    print(f"  figures      {n} PNGs -> docs/figures/")


def copy_pdf():
    src = os.path.join(REPORTS, REPORT_PDF)
    if not os.path.exists(src):
        raise SystemExit(f"ABORT: missing {src}")
    shutil.copy2(src, os.path.join(DOCS, "report.pdf"))
    print(f"  report.pdf   {REPORT_PDF} ({os.path.getsize(src)/1024:.0f} KB)")


def build_report():
    src = os.path.join(REPORTS, "report.html")
    s = open(src, encoding="utf-8").read()

    s = s.replace('src="../figures/', 'src="figures/')
    if "../" in s:
        raise SystemExit("ABORT: report.html still has a parent-relative path")

    s = s.replace("</style>", SCREEN_CSS + "</style>", 1)
    s = s.replace("<body>\n", '<body>\n<a class="backlink" href="./">← Project home</a>\n', 1)
    if 'class="backlink" href="./"' not in s:
        raise SystemExit("ABORT: could not insert the report back link")

    out = os.path.join(DOCS, "report.html")
    open(out, "w", encoding="utf-8").write(s)
    print(f"  report.html  {len(s):,} chars, figure paths rewritten")


def build_model_card():
    src = os.path.join(REPORTS, "MODEL_CARD.md")
    md = open(src, encoding="utf-8").read()
    shutil.copy2(src, os.path.join(DOCS, "MODEL_CARD.md"))

    body = mistune.create_markdown(plugins=["table", "strikethrough"])(md)
    lines = body.split("\n")
    if lines and lines[0].startswith("<h1>"):
        title = re.sub(r"</?h1>", "", lines[0])
        lines = lines[1:]
    else:
        title = "Model Card"
    page = MODEL_CARD_PAGE.format(title=title, body="\n".join(lines))

    out = os.path.join(DOCS, "model-card.html")
    open(out, "w", encoding="utf-8").write(page)
    print(f"  model-card   {body.count('<table>')} tables rendered -> docs/model-card.html")


def patch_notebook():
    """Add the home link to nbconvert's output, if it has been generated."""
    out = os.path.join(DOCS, "notebook.html")
    if not os.path.exists(out):
        print("  notebook     SKIP (docs/notebook.html not generated yet)")
        return
    s = open(out, encoding="utf-8").read()
    if "nb-backlink" in s:
        print("  notebook     already has the back link")
        return
    i = s.find("<body")
    if i == -1:
        raise SystemExit("ABORT: no <body> in docs/notebook.html")
    j = s.index(">", i) + 1
    s = s[:j] + "\n" + NB_BANNER + s[j:]
    open(out, "w", encoding="utf-8").write(s)
    print("  notebook     back link added")


def run():
    os.makedirs(DOCS, exist_ok=True)
    print("building docs/")
    copy_figures()
    copy_pdf()
    build_report()
    build_model_card()
    patch_notebook()
    print("done.")


if __name__ == "__main__":
    run()
