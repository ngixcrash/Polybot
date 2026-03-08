"""Append-only JSONL trade storage and JSON model state persistence."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
TRADES_FILE = DATA_DIR / "trades.jsonl"
MODEL_FILE = DATA_DIR / "indicator_weights.json"


class TradeStore:

    def __init__(self, path: Path = TRADES_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            log.error(f"Failed to persist trade: {e}")

    def update_last(self, updates: dict) -> None:
        """Merge updates into the last trade record (reads all, rewrites file)."""
        try:
            if not self.path.exists():
                return

            lines = self.path.read_text(encoding="utf-8").strip().split("\n")
            if not lines or not lines[-1].strip():
                return

            last_record = json.loads(lines[-1])
            last_record.update(updates)
            lines[-1] = json.dumps(last_record, default=str)

            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tmp.replace(self.path)
        except Exception as e:
            log.error(f"Failed to update last trade: {e}")

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            log.error(f"Failed to load trades: {e}")
        return records

    def load_recent(self, n: int = 200) -> list[dict]:
        all_records = self.load_all()
        return all_records[-n:]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0


class ModelStore:

    def __init__(self, path: Path = MODEL_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            tmp.replace(self.path)
            log.debug(f"Model state saved ({len(state)} keys)")
        except Exception as e:
            log.error(f"Failed to save model state: {e}")

    def load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                state = json.load(f)
            log.info(f"Loaded model state from {self.path}")
            return state
        except Exception as e:
            log.error(f"Failed to load model state: {e}")
            return None
