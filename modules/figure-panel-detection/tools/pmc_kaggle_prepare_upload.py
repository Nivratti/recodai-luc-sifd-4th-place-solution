#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional tqdm
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Kaggle API practical limits
TITLE_MIN, TITLE_MAX = 6, 50
SUBTITLE_MIN, SUBTITLE_MAX = 20, 80
SLUG_MIN, SLUG_MAX = 3, 50

MONTH_ALIASES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

CADENCE_CANON = {
    "weekly": "Weekly",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "halfyear": "Half-year",
    "half-year": "Half-year",
    "yearly": "Yearly",
    "annual": "Yearly",
}


@dataclass
class PeriodInfo:
    date_start: Optional[str]   # YYYY-MM-DD
    date_end: Optional[str]     # YYYY-MM-DD
    query_name: str             # base/topicv1/...
    period_label: str           # e.g. Weekly/Monthly/Quarterly/Custom range
    start_compact: Optional[str]  # YYYYMMDD (if available)
    end_compact: Optional[str]    # YYYYMMDD (if available)


@dataclass
class ScanStats:
    scanned_at_utc: str
    root: str
    papers_root: str
    batch_count: int
    paper_count: int
    figure_count: int
    papers_with_nxml: int
    papers_with_metadata_json: int
    papers_with_manifest_json: int
    image_ext_counts: Dict[str, int]
    license_type_counts: Dict[str, int]
    missing_metadata_papers: int
    missing_manifest_papers: int
    warnings: List[str]


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.2f}{u}" if u != "B" else f"{int(f)}B"
        f /= 1024.0
    return f"{f:.2f}TB"

def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total

def run_cmd_live(cmd: List[str], cwd: Path) -> int:
    print(f"\n[RUN] {' '.join(cmd)}")
    sys.stdout.flush()

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
    p.wait()
    return int(p.returncode)

def sanitize_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def len_ok(s: str, lo: int, hi: int) -> bool:
    return lo <= len(s) <= hi


def validate_or_fix_len(label: str, s: str, lo: int, hi: int, non_interactive: bool) -> str:
    if len_ok(s, lo, hi):
        return s
    if non_interactive:
        s2 = s[:hi].rstrip()
        if len(s2) < lo:
            s2 = (s2 + " " * lo)[:lo]
        return s2

    print(f"\n[WARN] {label} length={len(s)} but must be {lo}..{hi}.")
    while True:
        print(f"{label} current: {s}")
        newv = input("Enter to keep, or type replacement: ").strip()
        if newv:
            s = newv
        if len_ok(s, lo, hi):
            return s
        print(f"Still invalid length: {len(s)}")


def detect_cadence_from_path(root: Path) -> str:
    """
    Look at folder names in the path to detect cadence (weekly/monthly/quarterly/...).
    Returns a label like "Weekly" or "Custom range" if unknown.
    """
    for part in reversed(root.parts):
        key = part.strip().lower()
        if key in CADENCE_CANON:
            return CADENCE_CANON[key]
    return "Custom range"


def _fmt_date(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


def _compact(y: int, m: int, d: int) -> str:
    return f"{y:04d}{m:02d}{d:02d}"


def infer_period_from_root(root: Path) -> PeriodInfo:
    """
    Supports:
      1) YYYYMMDD_YYYYMMDD_query        e.g. 20251201_20251208_base
      2) YYYY-MM-DD_YYYY-MM-DD_query    e.g. 2025-12-01_2025-12-08_base
      3) YYYY-Mon_query or YYYY-Month_query or YYYY-MM_query
         e.g. 2025-Nov_base, 2025-11_base
      4) YYYYQ[1-4]_query or YYYY-Q4_query
         e.g. 2025Q4_base, 2025-Q4_topicv1
      5) YYYYH[1-2]_query or YYYY-H1_query
         e.g. 2025H2_base
      6) YYYY_query (only treated as yearly if cadence says Yearly)
         e.g. 2025_base in /yearly/
    """
    name = root.name.strip()
    cadence = detect_cadence_from_path(root)

    # 1) 20251201_20251208_base
    m = re.match(r"^(\d{8})_(\d{8})_([A-Za-z0-9\-]+)$", name)
    if m:
        d1, d2, q = m.group(1), m.group(2), m.group(3)
        y1, mo1, da1 = int(d1[0:4]), int(d1[4:6]), int(d1[6:8])
        y2, mo2, da2 = int(d2[0:4]), int(d2[4:6]), int(d2[6:8])
        return PeriodInfo(
            date_start=_fmt_date(y1, mo1, da1),
            date_end=_fmt_date(y2, mo2, da2),
            query_name=q,
            period_label=cadence,
            start_compact=_compact(y1, mo1, da1),
            end_compact=_compact(y2, mo2, da2),
        )

    # 2) 2025-12-01_2025-12-08_base
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})_(\d{4})-(\d{2})-(\d{2})_([A-Za-z0-9\-]+)$", name)
    if m:
        y1, mo1, da1, y2, mo2, da2, q = m.groups()
        y1, mo1, da1 = int(y1), int(mo1), int(da1)
        y2, mo2, da2 = int(y2), int(mo2), int(da2)
        return PeriodInfo(
            date_start=_fmt_date(y1, mo1, da1),
            date_end=_fmt_date(y2, mo2, da2),
            query_name=q,
            period_label=cadence,
            start_compact=_compact(y1, mo1, da1),
            end_compact=_compact(y2, mo2, da2),
        )

    # 3) Monthly: 2025-Nov_base or 2025-11_base
    m = re.match(r"^(\d{4})[-_](\d{1,2}|[A-Za-z]{3,9})[-_]?([A-Za-z0-9\-]+)$", name)
    if m:
        y_s, mon_s, q = m.groups()
        y = int(y_s)
        mon_key = mon_s.strip().lower()
        if mon_key.isdigit():
            mo = int(mon_key)
            if 1 <= mo <= 12:
                last_day = calendar.monthrange(y, mo)[1]
                return PeriodInfo(
                    date_start=_fmt_date(y, mo, 1),
                    date_end=_fmt_date(y, mo, last_day),
                    query_name=q,
                    period_label="Monthly" if cadence == "Custom range" else cadence,
                    start_compact=_compact(y, mo, 1),
                    end_compact=_compact(y, mo, last_day),
                )
        else:
            if mon_key in MONTH_ALIASES:
                mo = MONTH_ALIASES[mon_key]
                last_day = calendar.monthrange(y, mo)[1]
                return PeriodInfo(
                    date_start=_fmt_date(y, mo, 1),
                    date_end=_fmt_date(y, mo, last_day),
                    query_name=q,
                    period_label="Monthly" if cadence == "Custom range" else cadence,
                    start_compact=_compact(y, mo, 1),
                    end_compact=_compact(y, mo, last_day),
                )

    # 4) Quarter: 2025Q4_base or 2025-Q4_base
    m = re.match(r"^(\d{4})[-_]?Q([1-4])[-_]?([A-Za-z0-9\-]+)$", name, re.IGNORECASE)
    if m:
        y_s, qn_s, qname = m.groups()
        y = int(y_s)
        qn = int(qn_s)
        start_mo = 1 + (qn - 1) * 3
        end_mo = start_mo + 2
        end_day = calendar.monthrange(y, end_mo)[1]
        return PeriodInfo(
            date_start=_fmt_date(y, start_mo, 1),
            date_end=_fmt_date(y, end_mo, end_day),
            query_name=qname,
            period_label="Quarterly" if cadence == "Custom range" else cadence,
            start_compact=_compact(y, start_mo, 1),
            end_compact=_compact(y, end_mo, end_day),
        )

    # 5) Half-year: 2025H2_base or 2025-H1_topicv1
    m = re.match(r"^(\d{4})[-_]?H([1-2])[-_]?([A-Za-z0-9\-]+)$", name, re.IGNORECASE)
    if m:
        y_s, hn_s, qname = m.groups()
        y = int(y_s)
        hn = int(hn_s)
        start_mo = 1 if hn == 1 else 7
        end_mo = 6 if hn == 1 else 12
        end_day = calendar.monthrange(y, end_mo)[1]
        return PeriodInfo(
            date_start=_fmt_date(y, start_mo, 1),
            date_end=_fmt_date(y, end_mo, end_day),
            query_name=qname,
            period_label="Half-year" if cadence == "Custom range" else cadence,
            start_compact=_compact(y, start_mo, 1),
            end_compact=_compact(y, end_mo, end_day),
        )

    # 6) Yearly: 2025_base (only if cadence says Yearly)
    m = re.match(r"^(\d{4})[-_]?([A-Za-z0-9\-]+)$", name)
    if m and cadence == "Yearly":
        y_s, qname = m.groups()
        y = int(y_s)
        return PeriodInfo(
            date_start=_fmt_date(y, 1, 1),
            date_end=_fmt_date(y, 12, 31),
            query_name=qname,
            period_label="Yearly",
            start_compact=_compact(y, 1, 1),
            end_compact=_compact(y, 12, 31),
        )

    # Fallback: unknown/custom
    return PeriodInfo(
        date_start=None,
        date_end=None,
        query_name="base",
        period_label=cadence,
        start_compact=None,
        end_compact=None,
    )


def safe_read_json(p: Path) -> Optional[dict]:
    try:
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_paper_dirs(papers_root: Path):
    if not papers_root.exists():
        return
    for batch_dir in sorted([p for p in papers_root.iterdir() if p.is_dir() and p.name.startswith("batch_")]):
        for paper_dir in sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.upper().startswith("PMC")]):
            yield paper_dir.name, paper_dir


def scan_dataset(root: Path, papers_rel: str = "data/papers") -> ScanStats:
    papers_root = (root / papers_rel)
    warnings: List[str] = []

    if not papers_root.exists():
        warnings.append(f"Missing papers root: {papers_root.as_posix()}")

    batch_count = 0
    if papers_root.exists():
        batch_count = len([p for p in papers_root.iterdir() if p.is_dir() and p.name.startswith("batch_")])

    paper_count = 0
    figure_count = 0
    papers_with_nxml = 0
    papers_with_metadata_json = 0
    papers_with_manifest_json = 0
    missing_metadata = 0
    missing_manifest = 0

    ext_counts: Dict[str, int] = {}
    lic_counts: Dict[str, int] = {}

    paper_list = list(iter_paper_dirs(papers_root))
    it = tqdm(paper_list, desc="scan papers", unit="paper") if tqdm else paper_list

    for _, paper_dir in it:
        paper_count += 1
        files = list(paper_dir.iterdir())

        imgs = [p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        figure_count += len(imgs)
        for p in imgs:
            ext = p.suffix.lower().lstrip(".")
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        if any(p.is_file() and p.suffix.lower() == ".nxml" for p in files):
            papers_with_nxml += 1

        meta_path = paper_dir / "metadata.json"
        if meta_path.exists():
            papers_with_metadata_json += 1
            meta = safe_read_json(meta_path) or {}
            lic_type = None
            if isinstance(meta, dict):
                lic = meta.get("license")
                if isinstance(lic, dict):
                    lic_type = lic.get("license_type") or lic.get("type")
                lic_type = lic_type or meta.get("license_type")
            if lic_type:
                lic_counts[str(lic_type).strip()] = lic_counts.get(str(lic_type).strip(), 0) + 1
        else:
            missing_metadata += 1

        manifest_path = paper_dir / "figures_manifest.json"
        if manifest_path.exists():
            papers_with_manifest_json += 1
        else:
            missing_manifest += 1

    if paper_count == 0:
        warnings.append("No paper directories found under data/papers/batch_*/PMC*/")

    return ScanStats(
        scanned_at_utc=iso_utc_now(),
        root=str(root.resolve()),
        papers_root=str(papers_root.resolve()),
        batch_count=batch_count,
        paper_count=paper_count,
        figure_count=figure_count,
        papers_with_nxml=papers_with_nxml,
        papers_with_metadata_json=papers_with_metadata_json,
        papers_with_manifest_json=papers_with_manifest_json,
        image_ext_counts=dict(sorted(ext_counts.items(), key=lambda x: (-x[1], x[0]))),
        license_type_counts=dict(sorted(lic_counts.items(), key=lambda x: (-x[1], x[0]))),
        missing_metadata_papers=missing_metadata,
        missing_manifest_papers=missing_manifest,
        warnings=warnings,
    )


def propose_strings(p: PeriodInfo) -> Tuple[str, str, str, str, str]:
    """
    Returns: long_title, long_subtitle, slug, api_title, api_subtitle
    """
    q = p.query_name
    q_label = "Base" if q.lower() == "base" else q

    if p.date_start and p.date_end:
        long_title = f"Biomedical Figures from PMC OA — {p.period_label} ({p.date_start} to {p.date_end}) [{q_label}]"
        long_subtitle = f"Open-access biomedical figure images collected from PMC OA ({q.lower()} query), {p.period_label.lower()} snapshot."
    else:
        long_title = f"Biomedical Figures from PMC OA — {p.period_label} [{q_label}]"
        long_subtitle = f"Open-access biomedical figure images collected from PMC OA ({q.lower()} query)."

    # slug
    if p.start_compact and p.end_compact:
        slug = f"pmc-oa-biomed-figures-{p.start_compact}-{p.end_compact}-{q.lower()}"
    else:
        slug = f"pmc-oa-biomed-figures-{p.period_label.lower().replace(' ', '-')}-{q.lower()}"
    slug = sanitize_slug(slug)
    slug = validate_or_fix_len("Slug", slug, SLUG_MIN, SLUG_MAX, non_interactive=True)

    # API title/subtitle (must be shorter)
    if p.date_start and p.date_end:
        api_title = f"PMC OA Biomed Figures ({p.date_start}–{p.date_end[5:]})"
        api_subtitle = f"Open-access PMC OA figure images ({q.lower()}), {p.date_start} to {p.date_end}."
    else:
        api_title = "PMC OA Biomed Figures"
        api_subtitle = f"Open-access PMC OA figure images ({q.lower()})."

    api_title = validate_or_fix_len("API Title", api_title, TITLE_MIN, TITLE_MAX, non_interactive=True)
    api_subtitle = validate_or_fix_len("API Subtitle", api_subtitle, SUBTITLE_MIN, SUBTITLE_MAX, non_interactive=True)

    return long_title, long_subtitle, slug, api_title, api_subtitle


def read_kaggle_username() -> Optional[str]:
    u = os.environ.get("KAGGLE_USERNAME", "").strip()
    if u:
        return u
    p = Path.home() / ".config" / "kaggle" / "kaggle.json"
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            u2 = obj.get("username")
            if isinstance(u2, str) and u2.strip():
                return u2.strip()
        except Exception:
            pass
    return None


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme(root: Path, long_title: str, long_subtitle: str, p: PeriodInfo, stats: ScanStats, keywords: List[str]) -> Path:
    out = root / "README.md"
    lines: List[str] = []
    lines.append(f"# {long_title}")
    lines.append("")
    lines.append(long_subtitle)
    lines.append("")
    lines.append("## What to use")
    lines.append("- Use `data/papers/` for the actual figure images.")
    lines.append("- Other root files are provenance/index artifacts produced by the harvesting pipeline.")
    lines.append("")
    lines.append("## Period")
    lines.append(f"- Period label: **{p.period_label}**")
    lines.append(f"- Query: **{p.query_name}**")
    lines.append(f"- Date range: **{p.date_start or 'unknown'} → {p.date_end or 'unknown'}**")
    lines.append("")
    lines.append("## Quick stats")
    lines.append(f"- Papers: **{stats.paper_count}**")
    lines.append(f"- Figure images: **{stats.figure_count}**")
    lines.append("")
    lines.append("### Image extensions")
    for k, v in stats.image_ext_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Kaggle keywords")
    lines.append(f"- {', '.join(keywords)}")
    lines.append("")
    if stats.warnings:
        lines.append("## Warnings")
        for w in stats.warnings:
            lines.append(f"- {w}")
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare Kaggle metadata from a PMC OA dataset root and optionally upload.")
    ap.add_argument("--root", required=True, help="Dataset root folder (the folder you will upload).")
    ap.add_argument("--owner", default="", help="Kaggle username/org (default: read from ~/.kaggle/kaggle.json or env).")
    ap.add_argument("--papers-rel", default="data/papers", help="Relative path to papers folder under root.")
    ap.add_argument("--license", default="other", help="Kaggle license name, e.g. other, CC0-1.0, CC-BY-4.0")

    # Valid Kaggle keywords (your UI ones)
    ap.add_argument("--keywords", default="Biology,Computer Vision,Drugs and Medications,Image",
                    help="Comma-separated Kaggle keywords (must match Kaggle tag list).")

    # Overrides if inference can’t parse a custom folder name
    ap.add_argument("--date-start", default=None, help="Override start date YYYY-MM-DD")
    ap.add_argument("--date-end", default=None, help="Override end date YYYY-MM-DD")
    ap.add_argument("--query-name", default=None, help="Override query name (e.g., base/topicv1)")
    ap.add_argument("--period-label", default=None, help="Override period label (Weekly/Monthly/Quarterly/Custom/...)")

    ap.add_argument("--mode", choices=["prepare-only", "auto", "create", "version"], default="prepare-only")
    ap.add_argument("--message", default="Metadata update", help="Version message for Kaggle upload.")
    ap.add_argument("--non-interactive", action="store_true", help="Do not prompt; auto-accept generated fields.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    p = infer_period_from_root(root)

    # Apply overrides (for custom-range folder names)
    if args.date_start:
        p.date_start = args.date_start
    if args.date_end:
        p.date_end = args.date_end
    if args.query_name:
        p.query_name = args.query_name
    if args.period_label:
        p.period_label = args.period_label

    long_title, long_subtitle, slug, api_title, api_subtitle = propose_strings(p)

    # Interactive correction
    if not args.non_interactive:
        print("\n=== Generated from input path ===")
        print(f"Root:        {root}")
        print(f"Period:      {p.period_label}")
        print(f"Query:       {p.query_name}")
        print(f"Range:       {p.date_start} -> {p.date_end}")
        print(f"Slug:        {slug} (len={len(slug)})")
        print(f"API title:   {api_title} (len={len(api_title)})")
        print(f"API subtitle:{api_subtitle} (len={len(api_subtitle)})")

        slug_in = input("\nEnter new slug or press Enter to keep: ").strip()
        if slug_in:
            slug = sanitize_slug(slug_in)
        slug = validate_or_fix_len("Slug", slug, SLUG_MIN, SLUG_MAX, non_interactive=False)

        t_in = input("Enter new API title or press Enter to keep: ").strip()
        if t_in:
            api_title = t_in
        api_title = validate_or_fix_len("API Title", api_title, TITLE_MIN, TITLE_MAX, non_interactive=False)

        s_in = input("Enter new API subtitle or press Enter to keep: ").strip()
        if s_in:
            api_subtitle = s_in
        api_subtitle = validate_or_fix_len("API Subtitle", api_subtitle, SUBTITLE_MIN, SUBTITLE_MAX, non_interactive=False)

        ok = input("Proceed? [Y/n] ").strip().lower()
        if ok and ok not in ("y", "yes"):
            print("Cancelled.")
            return 1

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    # Scan + write files
    stats = scan_dataset(root, papers_rel=args.papers_rel)
    write_json(root / "summary.json", asdict(stats))
    write_readme(root, long_title, long_subtitle, p, stats, keywords)

    # Local overview (uploaded)
    write_json(root / "dataset_overview.json", {
        "dataset_title": long_title,
        "subtitle": long_subtitle,
        "slug_proposed": slug,
        "period_label": p.period_label,
        "date_range": {"start": p.date_start, "end": p.date_end},
        "query_name": p.query_name,
        "keywords": keywords,
        "generated_at_utc": iso_utc_now(),
        "root_layout": {
            "figures_root": "data/papers",
            "run_report": "data/pmc_oa_run_last.json",
            "discovery_report": "pmc_oa_discovery_last_run.json",
            "discovery_query": "pmc_oa_discovery_query_last.txt",
            "sqlite_db": "db.sqlite",
        },
    })

    # Kaggle dataset-metadata.json
    owner = args.owner.strip() or (read_kaggle_username() or "")
    if args.mode != "prepare-only" and not owner:
        raise RuntimeError("Owner required for upload. Use --owner or ensure ~/.kaggle/kaggle.json exists.")

    description = (
        f"# {long_title}\n\n"
        f"**Subtitle:** {long_subtitle}\n\n"
        f"**Source:** Europe PMC / PubMed Central Open Access (PMC OA)\n\n"
        f"**Payload:** `data/papers/`\n\n"
        f"**Provenance:** `pmc_oa_discovery_last_run.json`, `pmc_oa_discovery_query_last.txt`, `data/pmc_oa_run_last.json`\n\n"
        f"**Scan summary:** {stats.paper_count} papers, {stats.figure_count} figure images.\n"
    )

    write_json(root / "dataset-metadata.json", {
        "title": api_title,
        "subtitle": api_subtitle,
        "description": description,
        "id": f"{owner}/{slug}" if owner else "OWNER_REQUIRED/SLUG_REQUIRED",
        "licenses": [{"name": args.license.strip()}],
        "keywords": keywords,
    })

    print("\n[OK] Wrote: README.md, summary.json, dataset_overview.json, dataset-metadata.json")

    if args.mode == "prepare-only":
        print("[DONE] prepare-only mode.")
        return 0

    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI not found. Install: pip install -U kaggle")

    print("\n[STAGE] Metadata generation complete.")
    print("[STAGE] Starting Kaggle upload...")

    # Always use --dir-mode zip so folders like data/ upload
    print("\n[STAGE] Metadata generation complete.")
    print("[STAGE] Starting Kaggle upload...")

    # Helpful message: Kaggle will create data.zip when using --dir-mode zip
    data_dir = root / "data"
    if data_dir.exists():
        approx = _human_bytes(_dir_size_bytes(data_dir))
        print(f"[INFO] data/ folder detected (~{approx}). Kaggle will bundle it as data.zip (this can take time).")
    else:
        print("[INFO] No data/ folder found. Upload will be fast but may not include figures.")

    if args.mode in ("auto", "create"):
        print("[STAGE] kaggle datasets create (may take time during zipping/upload)...")
        code = run_cmd_live(["kaggle", "datasets", "create", "-p", str(root), "--dir-mode", "zip"], cwd=root)
        if code == 0:
            print("\n[OK] Created dataset successfully.")
            return 0
        if args.mode == "create":
            print("\n[ERROR] Create failed.")
            return code
        print("\n[WARN] Create failed; falling back to version upload...")

    print("[STAGE] kaggle datasets version (may take time during zipping/upload)...")
    code = run_cmd_live(
        ["kaggle", "datasets", "version", "-p", str(root), "-m", args.message, "--dir-mode", "zip"],
        cwd=root
    )
    if code != 0:
        print("\n[ERROR] Version upload failed.")
        return code

    print("\n[OK] Uploaded new version successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())