from html.parser import HTMLParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"

class SiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.references = []
        self.h1_count = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        element_id = attrs.get("id")
        if element_id:
            self.ids.append(element_id)
        if tag == "h1":
            self.h1_count += 1
        href = attrs.get("href")
        src = attrs.get("src")
        if href:
            self.references.append(href)
        if src:
            self.references.append(src)

def main():
    if not INDEX.exists():
        print("ERROR: portfolio-site/index.html is missing.")
        return 1

    parser = SiteParser()
    parser.feed(INDEX.read_text(encoding="utf-8"))
    errors = []

    duplicates = sorted({value for value in parser.ids if parser.ids.count(value) > 1})
    if duplicates:
        errors.append(f"Duplicate HTML id values: {', '.join(duplicates)}")

    if parser.h1_count != 1:
        errors.append(f"Expected exactly one h1, found {parser.h1_count}.")

    known_ids = set(parser.ids)
    for reference in parser.references:
        if reference.startswith("#"):
            target = reference[1:]
            if target and target not in known_ids:
                errors.append(f"Broken in-page link: {reference}")
            continue

        if reference.startswith(("https://", "http://", "mailto:", "tel:", "data:")):
            continue

        local_path = (ROOT / reference.split("?", 1)[0]).resolve()
        if not local_path.is_relative_to(ROOT.resolve()):
            errors.append(f"Local reference escapes portfolio-site: {reference}")
        elif not local_path.exists():
            errors.append(f"Missing local asset: {reference}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Portfolio checks passed: {len(parser.ids)} IDs, {len(parser.references)} local/external references, one h1.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
