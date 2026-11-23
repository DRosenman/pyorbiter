"""Tidy the reference pages that `quartodoc build` writes into reference/.

Run it right after `quartodoc build` and before `quarto render`:

    quartodoc build
    python _tidy_reference.py
    quarto render

It makes three small fixes to the generated Markdown:

* "See Also" entries, which numpydoc writes one per line, become a bulleted
  list of links instead of running together in one paragraph.
* Bullet lists inside a table cell (the columns of a returned DataFrame) are
  put on separate lines.
* Links in the index lose a `#fragment` that the rendered pages do not have.
"""

import re
from pathlib import Path

REFERENCE = Path(__file__).parent / "reference"
PAGES = {page.stem for page in REFERENCE.glob("*.qmd")}


def link(name: str) -> str:
    """A link to `name`'s reference page if it has one, else the bare name."""
    name = name.strip()
    target = name.removesuffix("()")
    for stem in (target, f"OrbitSystem.{target}"):
        if stem in PAGES:
            return f"[`{name}`]({stem}.qmd)"
    return f"`{name}`"


def see_also(match: re.Match) -> str:
    lines = []
    for line in match.group(2).strip().splitlines():
        name, sep, reason = line.partition(" : ")
        lines.append(f"- {link(name)}: {reason.strip()}" if sep else f"- {link(line)}")
    return match.group(1) + "\n".join(lines) + "\n\n"


def table_bullets(line: str) -> str:
    if not line.startswith("|"):
        return line
    return re.sub(r"\s+-\s+(?=``)", " <br>&bull; ", line)


for page in REFERENCE.glob("*.qmd"):
    text = page.read_text(encoding="utf-8")
    new = re.sub(r"(#+ See Also \{[^}]*\}\n\n)((?:[^\n#|][^\n]*\n)+)\n?", see_also, text)
    new = "\n".join(table_bullets(line) for line in new.split("\n"))
    if page.stem == "index":
        new = re.sub(r"(\.qmd)#pyorbiter\.[\w.]+\)", r"\1)", new)
    if new != text:
        page.write_text(new, encoding="utf-8")
