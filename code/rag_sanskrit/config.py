from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
# Paths
root: Path = Path(file).resolve().parents[2]
data_dir: Path = root / "data"
work_dir: Path = root / "code" / "artifacts"
chunks_path: Path = work_dir / "chunks.jsonl"
index_path: Path = work_dir / "index.npz"
