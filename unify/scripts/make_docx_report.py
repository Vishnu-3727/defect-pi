"""Render reports/ccp-report-unify-rul.md into the submission .docx.

    python scripts/make_docx_report.py
    python scripts/make_docx_report.py --demo      # self-check, writes to temp

The markdown file is the master text; this script only formats it. Body text is
Times New Roman 12 pt, 1.5 spacing, justified, which is what the department
template uses. Figure placeholders of the form

    *[Figure 5.1: Title]*

are replaced by the corresponding PNG from reports/figures/ when FIGURES maps
them, and left as a bracketed placeholder line otherwise - the reference report
does the same for its hand-drawn block diagrams.

This is a converter for THIS document, not a general markdown engine. It handles
the constructs the master file actually uses: headings, paragraphs, pipe tables,
fenced code, bullet/numbered lists, bold spans, and the figure placeholder. A
construct it does not recognise passes through as plain text rather than being
dropped, so nothing can silently vanish from a submitted report.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "ccp-report-unify-rul.md"
FIGDIR = ROOT / "reports" / "figures"

# The institute logo, extracted from the department's reference report where it
# is placed on the title page and on the bonafide certificate page. Its size
# there is 167.8 x 88.9 pt, which is 2.33 x 1.23 inches; reproduced exactly.
LOGO = ROOT / "reports" / "assets" / "cit_logo.png"
LOGO_WIDTH_IN = 2.33

# The wide letterhead that heads the institutional pages (vision, mission,
# programme outcomes) in the department template.
LETTERHEAD = ROOT / "reports" / "assets" / "cit-letterhead.png"
LETTERHEAD_WIDTH_IN = 6.0

# Layout markers, not printed titles: these two sections ARE the page, so the
# heading text must not appear on it. The page break still fires.
HIDDEN_HEADINGS = {
    "TITLE PAGE", "BONAFIDE CERTIFICATE PAGE",
    "INSTITUTION VISION AND MISSION", "DEPARTMENT VISION AND MISSION",
    "PROGRAM OUTCOMES", "PROGRAM SPECIFIC OUTCOMES AND OBJECTIVES",
}

# The two callout colours the department template uses: purple for the
# institutional boxes, green for the departmental ones.
BOX_COLOURS = {"box-purple": "7030A0", "box-green": "00A050"}

# Figure number in the report -> the measured PNG that belongs there. Figures
# absent from this map are conceptual block diagrams that have to be drawn by
# hand; they stay as placeholders exactly as in the department template.
FIGURES = {
    "4.1": "fig9_footprint.png",
    "5.1": "fig1_training_curve.png",
    "5.2": "fig2_pred_vs_true.png",
    "5.3": "fig3_error_hist.png",
    "5.4": "fig4_model_vs_baseline.png",
    "5.5": "fig5_pi_latency.png",
    "5.6": "fig6_realtime_budget.png",
    "5.7": "fig7_pi_live_trace.png",
    "5.8": "fig8_pi_service_latency.png",
    "5.9": "fig10_pi_dashboard.png",
}

# Sections that start on a fresh page, matched on the heading text.
PAGE_BREAK_BEFORE = (
    "BONAFIDE CERTIFICATE", "INSTITUTION VISION", "DEPARTMENT VISION",
    "PROGRAM OUTCOMES", "PROGRAM SPECIFIC OUTCOMES",
    "ACKNOWLEDGEMENT", "TABLE OF CONTENTS",
    "LIST OF TABLES", "ABSTRACT", "CHAPTER 1", "CHAPTER 2", "CHAPTER 3",
    "CHAPTER 4", "CHAPTER 5", "CHAPTER 6", "CHAPTER 7", "CHAPTER 8",
)

FIG_RE = re.compile(r"^\*\[Figure\s+([\d.]+):\s*(.+?)\]\*$")
LOGO_RE = re.compile(r"^\*\[LOGO\]\*$")
LETTERHEAD_RE = re.compile(r"^\*\[LETTERHEAD\]\*$")
DIV_RE = re.compile(r"^:::\s*([\w-]+)?\s*$")
CAPTION_RE = re.compile(r"^\*(Table|Figure)\s+([\d.]+):\s*(.+?)\*$")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


# ------------------------------------------------------------------ style ----
def setup(doc: Document):
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    pf = st.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for s in doc.sections:
        s.left_margin = Inches(1.25)
        s.right_margin = Inches(1.0)
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)


def rich(par, text: str):
    """Write text into a paragraph, honouring **bold** spans."""
    pos = 0
    for m in BOLD_RE.finditer(text):
        if m.start() > pos:
            par.add_run(text[pos:m.start()])
        par.add_run(m.group(1)).bold = True
        pos = m.end()
    if pos < len(text):
        par.add_run(text[pos:])
    return par


def heading(doc: Document, text: str, level: int):
    if any(text.upper().startswith(p) for p in PAGE_BREAK_BEFORE):
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    if text.upper() in HIDDEN_HEADINGS:
        return None
    p = doc.add_paragraph()
    p.paragraph_format.alignment = (WD_ALIGN_PARAGRAPH.CENTER if level <= 2
                                    else WD_ALIGN_PARAGRAPH.LEFT)
    p.paragraph_format.space_before = Pt(12)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt({1: 16, 2: 14, 3: 13}.get(level, 12))
    return p


def caption(doc: Document, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(11)


def code_block(doc: Document, lines: list[str]):
    for ln in lines:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.0
        pf.space_after = Pt(0)
        pf.left_indent = Inches(0.4)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(ln if ln.strip() else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(9)


def logo(doc: Document):
    """The institute logo, centred, at the size the reference report uses."""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(18)
    if LOGO.exists():
        p.add_run().add_picture(str(LOGO), width=Inches(LOGO_WIDTH_IN))
        return True
    r = p.add_run("[CHENNAI INSTITUTE OF TECHNOLOGY LOGO]")
    r.italic = True
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return False


def _borders(tbl, colour: str | None, size: int = 8):
    """Set every border of a table explicitly. colour=None removes them, which
    a table style alone does not guarantee across Word versions."""
    el = tbl._tbl.tblPr
    for old in el.findall(qn("w:tblBorders")):
        el.remove(old)
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge)
        if colour is None:
            e.set(qn("w:val"), "none")
            e.set(qn("w:sz"), "0")
        else:
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(size))
            e.set(qn("w:color"), colour)
        e.set(qn("w:space"), "0")
        b.append(e)
    el.append(b)


def letterhead(doc: Document):
    """The wide institute letterhead that heads each institutional page."""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    if LETTERHEAD.exists():
        p.add_run().add_picture(str(LETTERHEAD), width=Inches(LETTERHEAD_WIDTH_IN))
        return True
    r = p.add_run("[CHENNAI INSTITUTE OF TECHNOLOGY LETTERHEAD]")
    r.italic = True
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return False


def open_box(doc: Document, colour: str):
    """A single-cell coloured callout, as the template draws vision, mission,
    programme-outcome and objective statements."""
    t = doc.add_table(rows=1, cols=1)
    t.style = "Normal Table"
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _borders(t, colour, size=12)
    cell = t.rows[0].cells[0]
    cell.width = Inches(6.0)
    return t, cell


def close_box(tbl, cell):
    """Drop the empty paragraph python-docx puts in a new cell, and give the
    box a little breathing room below it."""
    paras = cell.paragraphs
    if len(paras) > 1 and not paras[0].text.strip():
        paras[0]._p.getparent().remove(paras[0]._p)
    for par in cell.paragraphs:
        par.paragraph_format.space_after = Pt(6)


def figure(doc: Document, num: str, title: str):
    png = FIGDIR / FIGURES[num] if num in FIGURES else None
    if png and png.exists():
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(png), width=Inches(5.7))
        caption(doc, "Figure %s: %s" % (num, title))
        return True
    # No measured image: leave the bracketed placeholder the template uses, so
    # a missing diagram is visible in the document instead of silently absent.
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("[Figure %s: %s]" % (num, title))
    r.italic = True
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return False


def table(doc: Document, rows: list[list[str]], plain: bool = False):
    """plain=True is the signature/examiner block on the certificate page: a
    borderless two-column layout, not a data table, so it takes no grid and no
    bold header row."""
    header, body = rows[0], rows[1:]
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Normal Table" if plain else "Table Grid"
    if plain:
        _borders(t, None)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for c, txt in zip(t.rows[0].cells, header):
        c.paragraphs[0].paragraph_format.alignment = (
            WD_ALIGN_PARAGRAPH.LEFT if plain else WD_ALIGN_PARAGRAPH.CENTER)
        c.paragraphs[0].paragraph_format.line_spacing = 1.0
        rich(c.paragraphs[0], txt)
        for r in c.paragraphs[0].runs:
            r.bold = not plain
            r.font.size = Pt(12 if plain else 11)
    for row in body:
        cells = t.add_row().cells
        for c, txt in zip(cells, row + [""] * (len(header) - len(row))):
            c.paragraphs[0].paragraph_format.line_spacing = 1.0
            rich(c.paragraphs[0], txt)
            for r in c.paragraphs[0].runs:
                r.font.size = Pt(12 if plain else 11)
    if plain:
        doc.add_paragraph().paragraph_format.space_after = Pt(12)


# ------------------------------------------------------------------ parse ----
def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def render(md: str, doc: Document):
    lines = md.splitlines()
    i, stats = 0, {"tables": 0, "figures": 0, "images": 0, "code": 0,
                   "headings": 0, "paragraphs": 0, "logos": 0}
    para: list[str] = []
    div = [None]           # current fenced-div mode: None | "center" | "plain" | "box-*"
    sect = [""]            # current level-2 section, for title-page sizing
    sink = [doc]           # where paragraphs land: the body, or an open box cell
    box = [None]           # (table, cell) while a callout box is open

    def flush():
        if not para:
            return
        text = " ".join(para).strip()
        para.clear()
        if not text:
            return
        m = CAPTION_RE.match(text)
        if m:
            caption(doc, "%s %s: %s" % (m.group(1), m.group(2), m.group(3)))
            return
        p = sink[0].add_paragraph()
        if div[0] == "center":
            # Title and certificate pages: centred, single-spaced, and the
            # title lines a size larger, matching the reference report.
            pf = p.paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.line_spacing = 1.15
            pf.space_after = Pt(10)
            rich(p, text)
            # On the title page the project title is set a size larger, as in
            # the reference. Everything else on both front pages stays 12 pt.
            if sect[0] == "TITLE PAGE" and text.startswith("**"):
                for r in p.runs:
                    r.font.size = Pt(14)
        else:
            rich(p, text)
        if div[0] == "compact":
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(4)
            for r in p.runs:
                r.font.size = Pt(10.5)
        stats["paragraphs"] += 1

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        m = DIV_RE.match(s)
        if m:
            flush()
            name = m.group(1)
            if box[0] is not None and name is None:
                close_box(*box[0])
                box[0], sink[0] = None, doc
            elif name in BOX_COLOURS:
                box[0] = open_box(doc, BOX_COLOURS[name])
                sink[0] = box[0][1]
                stats["tables"] += 1
            div[0] = name                # "::: name" opens, bare ":::" closes
            i += 1
            continue

        if LETTERHEAD_RE.match(s):
            flush()
            if letterhead(doc):
                stats["images"] += 1
            i += 1
            continue

        if LOGO_RE.match(s):
            flush()
            if logo(doc):
                stats["images"] += 1
            stats["logos"] += 1
            i += 1
            continue

        if s.startswith("```"):
            flush()
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            code_block(doc, block)
            stats["code"] += 1
            i += 1
            continue

        if s.startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            flush()
            rows = [split_row(s)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            table(doc, rows, plain=div[0] == "plain")
            stats["tables"] += 1
            continue

        m = FIG_RE.match(s)
        if m:
            flush()
            if figure(doc, m.group(1), m.group(2)):
                stats["images"] += 1
            stats["figures"] += 1
            i += 1
            continue

        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            text = s.lstrip("#").strip()
            if level == 2:
                sect[0] = text.upper()
            # The level-1 heading is the master file's own title. The document's
            # title is the title page, so it must not be printed above it.
            if level > 1:
                heading(doc, text, level)
                stats["headings"] += 1
            i += 1
            continue

        if s in ("---", ""):
            flush()
            i += 1
            continue

        if s.startswith("> "):
            i += 1                     # the master file's own build note
            continue

        # A list item or a numbered reference ends the previous paragraph.
        if re.match(r"^(\d+\.|[-*+])\s", s):
            flush()
            p = sink[0].add_paragraph()
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.first_line_indent = Inches(-0.35)
            body = [s]
            i += 1
            while i < len(lines) and lines[i].strip() and \
                    not re.match(r"^(\d+\.|[-*+]|#|\||>|```)", lines[i].strip()):
                body.append(lines[i].strip())
                i += 1
            rich(p, " ".join(body))
            if div[0] == "compact":
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(4)
                for r in p.runs:
                    r.font.size = Pt(10.5)
            stats["paragraphs"] += 1
            continue

        para.append(s)
        i += 1

    flush()
    return stats


def build(md: str, out: Path):
    doc = Document()
    setup(doc)
    stats = render(md, doc)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return stats


# ------------------------------------------------------------------- demo ----
def demo():
    """Self-check on a miniature document exercising every construct. The point
    is that nothing is silently dropped: counts must match what went in."""
    import shutil
    import tempfile

    global FIGURES
    saved = FIGURES
    FIGURES = {}                       # force the placeholder path
    md = "\n".join([
        "# TITLE",
        "",
        "## TITLE PAGE",
        "",
        "::: center",
        "",
        "**A CENTRED TITLE LINE**",
        "",
        "*[LOGO]*",
        "",
        ":::",
        "",
        "::: plain",
        "",
        "| LEFT | RIGHT |",
        "|---|---|",
        "",
        ":::",
        "",
        "## CHAPTER 1  INTRODUCTION",
        "",
        "Body text with a **bold** span that must survive.",
        "",
        "*[Figure 1.1: A Conceptual Diagram]*",
        "",
        "| A | B |",
        "|---|---|",
        "| 1 | 2 |",
        "| 3 | 4 |",
        "",
        "*Table 1.1: A Caption*",
        "",
        "```python",
        "x = 1",
        "```",
        "",
        "1. A numbered reference that wraps",
        "   onto a second line.",
        "",
    ])
    d = Path(tempfile.mkdtemp())
    try:
        out = d / "t.docx"
        stats = build(md, out)
        assert out.exists() and out.stat().st_size > 0
        # "TITLE PAGE" is a layout marker and must not be counted or printed.
        # Two level-2 headings are seen; the level-1 title is not counted at
        # all, and TITLE PAGE is counted but not printed (asserted below).
        assert stats["headings"] == 2, stats
        assert stats["tables"] == 2, stats
        assert stats["figures"] == 1, stats
        assert stats["logos"] == 1, stats
        # The real logo exists in the repo, so it is embedded; the figure is not.
        assert stats["images"] == (1 if LOGO.exists() else 0), stats
        assert stats["code"] == 1, stats

        doc = Document(str(out))
        assert len(doc.tables) == 2 and len(doc.tables[0].rows) == 1
        assert doc.tables[0].style.name != doc.tables[1].style.name, \
            "plain and grid tables must not share a style"
        text = "\n".join(p.text for p in doc.paragraphs)
        assert "TITLE PAGE" not in text, "layout marker heading leaked into the body"
        assert not text.lstrip().startswith("TITLE"),             "the master file's own H1 leaked above the title page"
        assert "bold" in text and "[Figure 1.1" in text and "x = 1" in text
        centred = [p for p in doc.paragraphs
                   if p.text.strip() == "A CENTRED TITLE LINE"]
        assert centred and centred[0].paragraph_format.alignment == \
            WD_ALIGN_PARAGRAPH.CENTER, "::: center did not centre its content"
        # The wrapped list item must be one paragraph, not two fragments.
        assert "onto a second line." in text
        assert any("A numbered reference that wraps onto a second line." in p.text
                   for p in doc.paragraphs), "list continuation was split"
        # A bold span must be a bold run, not literal asterisks.
        assert "**" not in text, "bold markers leaked into the document"
    finally:
        FIGURES = saved
        shutil.rmtree(d, ignore_errors=True)
    print("demo OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(ROOT / "reports" /
                                         "UNIFY-RUL_Project_Report.docx"))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()

    src = Path(a.src)
    if not src.exists():
        raise SystemExit("missing master text: %s" % src)
    stats = build(src.read_text(encoding="utf-8"), Path(a.out))
    missing = [n for n in sorted(FIGURES) if not (FIGDIR / FIGURES[n]).exists()]
    print("wrote %s" % a.out)
    print("  headings %(headings)d  paragraphs %(paragraphs)d  tables %(tables)d"
          "  figures %(figures)d (%(images)d embedded)  code blocks %(code)d"
          % stats)
    if missing:
        print("  WARNING: mapped figures not on disk: %s" % ", ".join(missing))
        print("  run: python scripts/make_report_figures.py --tag cmapss_full")


if __name__ == "__main__":
    sys.exit(main())
