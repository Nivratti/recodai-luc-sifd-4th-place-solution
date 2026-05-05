#!/usr/bin/env python3
"""
beautify_json_recursive_tqdm.py

Recursively pretty-print JSON files under a root directory and update in-place.

Defaults:
- tqdm progress bar
- indent=4
- ensure_ascii=False
- newline at EOF
- on any error: show full traceback and STOP (fail-fast)

Options:
- --on-error {stop,continue}   (default: stop)
- --backup                     create *.bak (does not overwrite existing backup)
- --dry-run                    don't write, just report count
- --sort-keys                  stable ordering
- --glob                       filename pattern (default: *.json)
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from tqdm import tqdm


def beautify_one(path: Path, *, indent: int, sort_keys: bool) -> tuple[bool, str]:
    """
    Returns (changed, output_or_status).
    If changed=True, output_or_status is the pretty JSON text to write.
    If changed=False, output_or_status is a status string: "NO_CHANGE".
    Raises on any read/parse issues.
    """
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)

    pretty = json.dumps(
        obj,
        indent=indent,
        ensure_ascii=False,
        sort_keys=sort_keys,
    ) + "\n"

    if pretty == raw:
        return False, "NO_CHANGE"
    return True, pretty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory to scan recursively")
    ap.add_argument("--indent", type=int, default=4, help="Indent spaces (default: 4)")
    ap.add_argument("--sort-keys", action="store_true", help="Sort keys in output JSON")
    ap.add_argument("--backup", action="store_true", help="Create .bak backup before overwriting")
    ap.add_argument("--dry-run", action="store_true", help="Do not write; only count changes")
    ap.add_argument("--glob", default="*.json", help="Filename glob (default: *.json)")
    ap.add_argument(
        "--on-error",
        choices=["stop", "continue"],
        default="stop",
        help="Error policy (default: stop with full traceback)",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    paths = sorted(root.rglob(args.glob))
    total = len(paths)
    if total == 0:
        print(f"[INFO] No files matched {args.glob} under {root}")
        return

    changed_count = 0

    pbar = tqdm(paths, total=total, desc="Beautifying JSON", unit="file", dynamic_ncols=True)
    for p in pbar:
        try:
            changed, out = beautify_one(p, indent=args.indent, sort_keys=args.sort_keys)
            if not changed:
                continue

            changed_count += 1
            pbar.set_postfix_str(f"updated={changed_count}")

            if args.dry_run:
                continue

            if args.backup:
                bak = p.with_suffix(p.suffix + ".bak")
                if not bak.exists():
                    bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

            p.write_text(out, encoding="utf-8")

        except Exception:
            # Full traceback required
            print("\n" + "=" * 80)
            print(f"[ERROR] Failed on file: {p}")
            print("=" * 80)
            traceback.print_exc()

            if args.on_error == "continue":
                continue
            raise  # stop

    print("\n==== Summary ====")
    print(f"Scanned : {total}")
    print(f"Updated : {changed_count}")
    print(f"Dry-run : {args.dry_run}")
    print(f"Backup  : {args.backup}")


if __name__ == "__main__":
    main()
