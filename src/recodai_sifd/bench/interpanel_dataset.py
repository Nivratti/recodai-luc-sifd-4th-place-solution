from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from .types import PairExample


@dataclass(frozen=True)
class InterpanelDataset:
    """Loads the inter-panel pair dataset from the exported folder layout.

    Expected layout (as in your dataset doc):
      - tasks/interpanel/match/.../A.png, B.png, A_mask.png, B_mask.png, meta.json
      - tasks/interpanel/no_match/.../A.png, B.png, meta.json

    We intentionally prefer directory scanning over CSV parsing because it is
    robust to column changes and works with materialize=copy/link.

    Args:
        tasks_root: path to ./.../tasks
        strict: if True, require A.png and B.png for each pair_dir and skip otherwise
    """
    tasks_root: Path
    strict: bool = True

    def interpanel_root(self) -> Path:
        return self.tasks_root / "interpanel"

    def _iter_pair_dirs(self, subset: str) -> Iterator[Path]:
        base = self.interpanel_root() / subset
        if not base.exists():
            return
        # pair dirs are leaf directories containing A.png and B.png
        for a_path in base.rglob("A.png"):
            pair_dir = a_path.parent
            b_path = pair_dir / "B.png"
            if self.strict and (not b_path.exists()):
                continue
            yield pair_dir

    def iter_examples(self, include_match: bool = True, include_no_match: bool = True) -> Iterator[PairExample]:
        if include_match:
            yield from self._iter_examples_for_subset("match", label=1)
        if include_no_match:
            yield from self._iter_examples_for_subset("no_match", label=0)

    def _iter_examples_for_subset(self, subset: str, *, label: int) -> Iterator[PairExample]:
        for pair_dir in self._iter_pair_dirs(subset):
            a_path = pair_dir / "A.png"
            b_path = pair_dir / "B.png"
            if self.strict and (not a_path.exists() or not b_path.exists()):
                continue

            a_mask = pair_dir / "A_mask.png"
            b_mask = pair_dir / "B_mask.png"
            meta_path = pair_dir / "meta.json"

            meta = None
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = None

            # stable-ish id (relative to tasks root)
            rel = pair_dir.relative_to(self.tasks_root).as_posix()
            pair_id = rel.replace("/", "__")

            yield PairExample(
                pair_id=pair_id,
                label=label,
                a_path=a_path,
                b_path=b_path,
                a_mask_path=a_mask if a_mask.exists() else None,
                b_mask_path=b_mask if b_mask.exists() else None,
                meta_path=meta_path if meta_path.exists() else None,
                meta=meta,
            )

    def list_examples(self, limit: Optional[int] = None, shuffle: bool = False, seed: int = 0,
                      include_match: bool = True, include_no_match: bool = True) -> List[PairExample]:
        ex = list(self.iter_examples(include_match=include_match, include_no_match=include_no_match))
        if shuffle:
            import random
            rng = random.Random(seed)
            rng.shuffle(ex)
        if limit is not None:
            ex = ex[: int(limit)]
        return ex
