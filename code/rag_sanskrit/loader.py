from future import annotations
from pathlib import Path
from typing import Dict, List
from pypdf import PdfReader
import re

_WS_RE = re.compile(r"[ \t]+")

def normalize_text(text: str) -> str:
# why: unify punctuation & whitespace for stable downstream behavior
text = text.replace("\u0964", "।").replace("\u0965", "॥")
text = re.sub(r"\r\n|\r", "\n", text)
text = "\n".join(_WS_RE.sub(" ", ln).strip() for ln in text.split("\n"))
return re.sub(r"\n{3,}", "\n\n", text)

def _read_txt(p: Path) -> str:
return p.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(p: Path) -> str:
reader = PdfReader(str(p))
return "\n".join((pg.extract_text() or "") for pg in reader.pages)

def load_documents(data_dir: Path) -> List[Dict]:
docs: List[Dict] = []
for p in sorted(data_dir.rglob("*")):
if p.is_dir():
continue
if p.suffix.lower() == ".txt":
raw = _read_txt(p)
elif p.suffix.lower() == ".pdf":
raw = _read_pdf(p)
else:
continue
text = normalize_text(raw)
if text.strip():
docs.append({"path": str(p), "text": text})
return docs
