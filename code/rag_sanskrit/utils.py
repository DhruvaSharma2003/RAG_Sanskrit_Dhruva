import logging, os
from typing import Optional

def setup_logging(level: int = logging.INFO) -> None:
fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(format=fmt, level=level)

def enforce_cpu_only() -> None:
# why: assignment requires CPU-only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_logger(name: Optional[str] = None) -> logging.Logger:
return logging.getLogger(name or "rag_sanskrit")
