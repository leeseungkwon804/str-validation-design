"""Manuscript_v2.md -> Manuscript_v2.docx, in the JTAER/MDPI layout.
Citation keys [@doi] are numbered in order of appearance and a reference list is written at the end.
Figures are inserted from figs/ as PNGs, with the caption and the source file name."""
import json, re, os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

MS = "./ms/Manuscript_v2.md"
FIG = "./figs"
refs = json.load(open("./ms/refs_v2.json"))
txt = open(MS).read()

order, num = [], {}
def cite_repl(m):
    out = []
    for k in [x.strip().lstrip("@") for x in m.group(1).split(";")]:
        if k not in num:
            order.append(k); num[k] = len(order)
        out.append(str(num[k]))
    return "[" + ",".join(out) + "]"

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Palatino Linotype"; st.font.size = Pt(10)
st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.15

def para(text, **kw):
    p = doc.add_paragraph(); run_md(p, text); return p

BOLD = re.compile(r"\*\*(.+?)\*\*")
def run_md(p, text):
    text = re.sub(r"\[@([^\]]+)\]", cite_repl, text)
    i = 0
    for m in BOLD.finditer(text):
        if m.start() > i: p.add_run(text[i:m.start()])
        p.add_run(m.group(1)).bold = True
        i = m.end()
    p.add_run(text[i:])

lines = txt.split("\n"); i = 0
while i < len(lines):
    ln = lines[i]
    if not ln.strip(): i += 1; continue
    if ln.startswith("TITLE:"):
        p = doc.add_paragraph(); r = p.add_run(ln[6:].strip())
        r.bold = True; r.font.size = Pt(16); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(12)
    elif ln.startswith("ABSTRACT:") or ln.startswith("KEYWORDS:"):
        lab, body = ln.split(":", 1)
        p = doc.add_paragraph(); p.add_run(lab.capitalize() + ": ").bold = True
        run_md(p, body.strip())
    elif ln.startswith("BACK:"):
        body = ln[5:].strip()
        p = doc.add_paragraph()
        if ":" in body[:60]:
            lab, rest = body.split(":", 1); p.add_run(lab + ": ").bold = True; run_md(p, rest.strip())
        else: run_md(p, body)
    elif ln.startswith("### "): doc.add_heading(re.sub(r"^###\s*", "", ln), level=2)
    elif ln.startswith("## "):  doc.add_heading(re.sub(r"^##\s*", "", ln), level=1)
    elif ln.startswith("FIGURE:"):
        key, cap = ln[7:].split("|", 1)
        key = key.strip(); path = f"{FIG}/{key}.png"
        if os.path.exists(path):
            doc.add_picture(path, width=Inches(6.2))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            q = doc.add_paragraph(); q.add_run(f"[FIGURE HERE: {key}.png]").bold = True
        p = doc.add_paragraph(); r = p.add_run(cap.strip()); r.font.size = Pt(9)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        q = doc.add_paragraph(); r = q.add_run(f"Source file: figs/{key}.png (and .pdf)")
        r.font.size = Pt(8); r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    elif ln.startswith("TABLE:"):
        p = doc.add_paragraph(); r = p.add_run(ln[6:].strip()); r.bold = True; r.font.size = Pt(9)
        rows = []
        j = i + 1
        while j < len(lines) and "|" in lines[j]:
            rows.append([c.strip() for c in lines[j].split("|")]); j += 1
        w = max(len(r_) for r_ in rows)
        t = doc.add_table(rows=len(rows), cols=w); t.style = "Table Grid"
        for a, r_ in enumerate(rows):
            for b in range(w):
                cell = t.cell(a, b); cell.text = ""
                run = cell.paragraphs[0].add_run(r_[b] if b < len(r_) else "")
                run.font.size = Pt(8)
                if a == 0: run.bold = True
        doc.add_paragraph()
        i = j; continue
    else:
        para(ln)
    i += 1

doc.add_heading("References", level=1)
def fmt(k):
    v = refs[k]
    au = v.get("authors") or []
    names = []
    for a in au[:6]:
        fam = (a[0] or "").strip(); giv = (a[1] if len(a) > 1 else "") or ""
        ini = " ".join(x[0] + "." for x in re.split(r"[\s\-]+", giv) if x)
        names.append(f"{fam}, {ini}".strip().rstrip(","))
    names = [n for n in names if n and n != ","]
    who = "; ".join(names) + ("; et al." if len(au) > 6 else "")
    bits = [who] if who else []
    bits.append(v.get("title", "").rstrip(".") + ".")
    j = v.get("short") or v.get("journal") or ""
    tail = ""
    if j and j != "?": tail += f"{j} "
    if v.get("year"): tail += f"{v['year']}"
    if v.get("volume"): tail += f", {v['volume']}"
    if v.get("page"): tail += f", {v['page']}"
    if tail.strip(): bits.append(tail.strip() + ".")
    if v.get("doi") and not str(v["doi"]).startswith(("reg", "ec", "ep", "inside", "fr_", "it_", "es_", "gr_", "pt_", "nl_", "be_", "at_", "ie_", "hu_")):
        bits.append(f"https://doi.org/{v['doi']}")
    if v.get("url"): bits.append(v["url"])
    return " ".join(bits)
for k in order:
    p = doc.add_paragraph(); r = p.add_run(f"{num[k]}. {fmt(k)}"); r.font.size = Pt(9)
    p.paragraph_format.space_after = Pt(3)

doc.save("./ms/Manuscript_v2.docx")
print(f"saved: ms/Manuscript_v2.docx, citations {len(order)}, tables {txt.count('TABLE:')}, figures {txt.count('FIGURE:')}")
