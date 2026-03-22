"""
Phase 1 — Corpus cleaning & chunking
Inputs:
  data/raw/react-native.txt   (llms.txt)
  data/raw/react-navigation.txt (llms.txt)
  data/raw/expo_git/          (mdx files)
  data/raw/reanimated_git/    (mdx files)
Outputs:
  data/processed/<lib>.jsonl  (chunks ready for RAG)
"""

import json
import re
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(exist_ok=True)

CHUNK_SIZE = 1500   # characters
CHUNK_OVERLAP = 200

# Files/dirs to skip
SKIP_PATTERNS = [
    "changelog", "migration", "upgrade", "release-note",
    "contributing", "license", "blog", "_shared", "_category_",
    "CHANGELOG", "versioned_docs/version-5", "versioned_docs/version-6",
]


def should_skip(path: Path) -> bool:
    p = str(path).lower()
    return any(pattern.lower() in p for pattern in SKIP_PATTERNS)


def clean_mdx(text: str) -> str:
    # Remove frontmatter
    text = re.sub(r"^---[\s\S]*?---\n", "", text, flags=re.MULTILINE)
    # Remove import statements
    text = re.sub(r"^import .+\n", "", text, flags=re.MULTILINE)
    # Remove JSX component tags (e.g. <APIBox>, <Callout>)
    text = re.sub(r"<[A-Z][a-zA-Z]+[^>]*>", "", text)
    text = re.sub(r"</[A-Z][a-zA-Z]+>", "", text)
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Collapse 3+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, source: str) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"source": source, "text": chunk.strip()})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def process_llms_txt(path: Path, lib: str):
    print(f"\n📄 Processing llms.txt: {lib}")
    text = path.read_text(encoding="utf-8")
    # Split on horizontal rules (---) which separate sections in llms.txt
    sections = re.split(r"\n-{3,}\n", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) > 100:  # skip trivially short sections
            chunks.extend(chunk_text(section, source=f"{lib}/llms.txt"))
    write_jsonl(chunks, lib)
    print(f"  → {len(chunks)} chunks")


def process_git_docs(folder: Path, lib: str):
    print(f"\n📁 Processing git docs: {lib}")
    files = [
        f for f in folder.rglob("*.md*")
        if not should_skip(f)
    ]
    chunks = []
    for f in files:
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_mdx(raw)
            if len(cleaned) < 100:
                continue
            rel = str(f.relative_to(folder))
            chunks.extend(chunk_text(cleaned, source=f"{lib}/{rel}"))
        except Exception as e:
            print(f"  ⚠️  {f.name}: {e}")
    write_jsonl(chunks, lib)
    print(f"  → {len(chunks)} chunks from {len(files)} files")


def write_jsonl(chunks: list[dict], lib: str):
    out = PROCESSED / f"{lib}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # llms.txt sources
    process_llms_txt(RAW / "react-native.txt", "react-native")
    process_llms_txt(RAW / "react-navigation.txt", "react-navigation")

    # git sparse-checkout sources
    process_git_docs(RAW / "expo_git" / "docs" / "pages", "expo")
    process_git_docs(RAW / "reanimated_git" / "docs" / "docs-reanimated" / "docs", "reanimated")

    # Summary
    print("\n✅ Corpus processed:")
    total = 0
    for f in sorted(PROCESSED.glob("*.jsonl")):
        count = sum(1 for _ in f.open())
        total += count
        print(f"  {f.name}: {count} chunks")
    print(f"  TOTAL: {total} chunks")
