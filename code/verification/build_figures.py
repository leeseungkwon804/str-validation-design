"""Reads the figure-number-to-file mapping from the manuscript's FIGURE lines and copies each to figures/FigureN.png.
The mapping is read from the manuscript itself, so a figure number cannot drift from its content."""
import re, shutil, os, json
from PIL import Image
MS = "./ms/Manuscript_v2.md"; FIG = "./figs"
OUT = "./pkg/figures"; os.makedirs(f"{OUT}/pdf", exist_ok=True)
rows = []
for m in re.finditer(r"^FIGURE:\s*(\S+)\s*\|\s*Figure\s+(\d+)\.\s*(.+)$", open(MS).read(), flags=re.M):
    key, n, cap = m.group(1), int(m.group(2)), m.group(3)
    src = f"{FIG}/{key}.png"
    dst = f"{OUT}/Figure{n}.png"
    shutil.copy2(src, dst); shutil.copy2(f"{FIG}/{key}.pdf", f"{OUT}/pdf/Figure{n}.pdf")
    im = Image.open(dst); dpi = im.info.get("dpi", (0, 0))[0]
    rows.append({"figure": n, "file": f"Figure{n}.png", "source": f"figs/{key}.png",
                 "px": f"{im.size[0]}x{im.size[1]}", "dpi": round(float(dpi)),
                 "caption": cap[:90]})
rows.sort(key=lambda r: r["figure"])
json.dump(rows, open(f"{OUT}/figure_manifest.json", "w"), indent=1, ensure_ascii=False)
assert [r["figure"] for r in rows] == list(range(1, len(rows) + 1)), "figure numbers are not consecutive"
for r in rows:
    print(f"  Figure{r['figure']}.png ← {r['source']:42s} {r['px']:>11s} {r['dpi']}dpi")
print(f"copied {len(rows)} figures (vector copies in figures/pdf/)")
