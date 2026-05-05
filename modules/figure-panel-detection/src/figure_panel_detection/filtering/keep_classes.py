from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np


def norm_key(s: str) -> str:
    return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum())


def parse_keep_classes_tokens(tokens: Optional[List[str]], names_map: Dict[int, str]) -> Optional[List[int]]:
    """
    Parse keep-class tokens that may be ids or names (case/space-insensitive).

    - Numeric tokens always work (even if names_map is empty).
    - Name tokens require names_map (use --names JSON/YAML).
    """
    if not tokens:
        return None

    flat: List[str] = []
    for t in tokens:
        for part in str(t).split(","):
            part = part.strip()
            if part:
                flat.append(part)

    # If any token is a name (non-digit), require names_map
    has_name_tokens = any(not t.isdigit() for t in flat)
    if has_name_tokens and not names_map:
        raise SystemExit(
            "[ERR] --keep-classes includes class names, but no class-name mapping is available.\n"
            "      Provide --names <path_to_names.json|yaml> or use numeric class ids.\n"
            "      Example names.json: {\"0\":\"Blots\",\"1\":\"Microscopy\",\"2\":\"Graphs\",\"3\":\"Flow Cytometry\"}\n"
        )

    name_to_id: Dict[str, int] = {norm_key(v): int(k) for k, v in names_map.items()}

    out_ids: List[int] = []
    unknown: List[str] = []
    for t in flat:
        if t.isdigit():
            out_ids.append(int(t))
            continue
        k = norm_key(t)
        if k in name_to_id:
            out_ids.append(name_to_id[k])
        else:
            unknown.append(t)

    if unknown:
        available = ", ".join([f"{k}:{v}" for k, v in sorted(names_map.items(), key=lambda x: x[0])])
        raise SystemExit(
            f"[ERR] Unknown class name(s): {unknown}\n"
            f"      Available model classes: {available}\n"
            f"      Tip: use quotes for spaces, e.g. --keep-classes \"Flow Cytometry\""
        )

    # unique preserve order
    seen: Set[int] = set()
    uniq: List[int] = []
    for x in out_ids:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def filter_by_class_ids(det: Optional[np.ndarray], keep_ids: Set[int]) -> Optional[np.ndarray]:
    if det is None or len(det) == 0:
        return det
    cls = det[:, 5].astype(np.int64)
    m = np.isin(cls, np.array(sorted(keep_ids), dtype=np.int64))
    return det[m]
