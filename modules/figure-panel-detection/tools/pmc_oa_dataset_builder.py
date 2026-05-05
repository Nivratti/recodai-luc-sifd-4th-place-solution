#!/usr/bin/env python3
#!/usr/bin/env python3
"""
PMC Open Access Dataset Builder — Resumable (Europe PMC discovery + PMC OA packages → XML + figures)

Purpose
-------
Build a local dataset of scientific paper assets from the PMC Open Access (OA) pipeline:

1) Discover candidate papers via Europe PMC REST search for a date range.
2) Store discovered PMCIDs + metadata into SQLite (resumable index).
3) For each pending PMCID:
   - Use PMC OA Web Service (oa.fcgi) to find/download the OA .tar.gz package
   - Extract the JATS XML (.nxml/.xml)
   - Parse figure references (<fig> -> <graphic xlink:href="...">)
   - Extract the referenced image files from the package (skipping thumbnails)
   - Write per-paper inspection files (metadata + figures manifest)
   - Track status + attempts in SQLite so runs can resume safely.

This script is designed for bulk downloads at scale (100s–10,000s papers) with
repeatable folder layout, robust error handling, and clear run reports.

Important constraints
---------------------
- The run step downloads via PMC OA Web Service, which requires a PMCID and OA package availability.
  Some records that look "open" in search results may still fail in oa.fcgi with an
  idIsNotOpenAccess error (meaning not in the OA subset); these are skipped as SKIPPED_NOT_OA.
- Discovery may search across any Europe PMC sources, but only records with a PMCID
  are stored (because PMC OA packages require PMCID).

Commands
--------
1) discover
   Discover PMCIDs for a date range and store/upsert into SQLite.

2) run
   Download + extract OA packages for pending PMCIDs (resumable), build per-paper dataset folders.

3) export
   Export summary CSVs (licenses.csv, figures.csv) from SQLite.

Quick start
-----------
# 1) Discover (basic query: OA + date + license; excludes reviews by default)
python pmc_oa_dataset_builder.py discover \
  --start 2025-12-01 --end 2025-12-08 \
  --db ./out/db.sqlite

# 2) Download + extract into default out dir (auto: <db_parent>/data)
python pmc_oa_dataset_builder.py run \
  --db ./out/db.sqlite

# 3) Export CSVs
python pmc_oa_dataset_builder.py export \
  --db ./out/db.sqlite --out ./out/reports

Discovery: query modes (profiles)
---------------------------------
Discovery supports multiple "query modes" so you can evolve your query over time without
rewriting code. The mode controls how the final query string is constructed.

--query-mode base
  - Minimal: OPEN_ACCESS:y + date range + allowed licenses + optional NOT review

--query-mode topic_v1
  - Adds a built-in topic clause (e.g., microscopy/blots keywords) on top of base

--query-mode custom_append
  - Final query = (base) AND (<your custom clause>)
  - Best for iterative changes (you keep base safety filters)

--query-mode custom_raw
  - Final query = exactly what you provide (no auto-filters added)
  - Use when you fully control the query yourself

Custom query inputs:
- --custom-query "...."         (inline string)
- --custom-query-file path.txt  (recommended; avoids shell quoting)

Include reviews (optional)
--------------------------
By default, discovery excludes review articles using a PUB_TYPE filter.
To include reviews, pass:
  --include-reviews

Reproducibility outputs (discover)
----------------------------------
Discovery writes TWO artifacts near the DB:

1) pmc_oa_discovery_query_last.txt
   - Plain text query (easy copy/paste into Europe PMC UI)

2) pmc_oa_discovery_last_run.json
   - JSON report with query, date range, discovery URLs, counts, and args

Run step outputs (dataset layout)
---------------------------------
The run step writes under <out_root>/ by default:
- raw_packages/                 downloaded .tar.gz files (optional delete)
- <paper_root>/...          per-paper extracted folders

You control per-paper folder layout with:

--layout flat
  <out_root>/<paper_root>/<PMCID>/

--layout batched  (DEFAULT)
  <out_root>/<paper_root>/batch_00000/<PMCID>/
  <out_root>/<paper_root>/batch_00001/<PMCID>/
  ...
  Batch size controlled by --batch-size (default 100).
  Batching is stable because each PMCID is assigned a persistent seq in the DB.

Per-paper contents
------------------
Inside each PMCID folder you typically get:
- *.nxml / *.xml                extracted JATS XML
- (image files...)              extracted figure image files referenced in XML
- metadata.json                 consolidated metadata (article + license + figures table)
- figures_manifest.json         inspection manifest: for each <fig>/<graphic> href, shows:
                               - extracted / skipped_thumbnail / missing_in_package / no_href
                               - also lists images present on disk but not referenced by <fig>

How to check “what images got filtered out”
-------------------------------------------
Open:
  <paper_folder>/figures_manifest.json

This file tells you:
- which hrefs existed in XML
- which hrefs resolved to files
- which were skipped as thumbnails
- which were missing in the OA package
- which image files exist but are unreferenced by <fig>

Delete packages to save disk
----------------------------
Large runs can consume huge disk space due to stored .tar.gz files. You can delete them
after a paper finishes extraction:

  --delete-raw-packages

Behavior:
- Deletes <out_root>/raw_packages/<PMCID>.tar.gz only after successful processing
  (DONE or DONE_NO_IMAGES). Failures keep the archive for debugging.

Resumability and statuses
-------------------------
SQLite tracks each PMCID with status + attempts:
- NEW                discovered, not processed yet
- DOWNLOADING         in progress
- EXTRACTING          in progress
- DONE                extracted and saved at least one image
- DONE_NO_IMAGES      processed, but no images saved (still keeps XML + manifests)
- SKIPPED_NOT_OA      oa.fcgi returned idIsNotOpenAccess
- SKIPPED_NO_TGZ      no tgz link / other oa.fcgi issue
- FAILED              error occurred; can retry on next run (attempts increment)

You can limit processing per run:
  --limit N

and cap retry attempts:
  --max-attempts N

Progress bar stats
------------------
--pbar-stats basic | all | none
- basic: done/failed/skip_not_oa/skip_no_tgz
- all: all counters
- none: no postfix stats

Output directory rules
----------------------
run:
- If --out is not provided, out_root defaults to:
    <db_parent>/data
- If --out is provided, it is used directly.

Folder naming recommendation
----------------------------
Use a neutral per-paper root name like:
- papers/   (recommended; includes XML + images + json)
- pmc_papers/

Avoid naming it "figures/" because the folder contains more than figures.

Common examples
---------------
# Discover with built-in microscopy/blot query
python pmc_oa_dataset_builder.py discover \
  --start 2025-12-01 --end 2025-12-08 \
  --db ./out/db.sqlite \
  --query-mode topic_v1

# Discover using your own clause appended to base filters
python pmc_oa_dataset_builder.py discover \
  --start 2025-12-01 --end 2025-12-08 \
  --db ./out/db.sqlite \
  --query-mode custom_append \
  --custom-query '(microscop* OR "western blot") AND (fig OR figure)'

# Run with batched layout: 200 papers per batch folder, delete tgz after extraction
python pmc_oa_dataset_builder.py run \
  --db ./out/db.sqlite \
  --out ./out/data \
  --layout batched --batch-size 200 \
  --paper-root papers \
  --delete-raw-packages

Troubleshooting
---------------
- "idIsNotOpenAccess" from oa.fcgi:
  The article may exist in PMC but not in the OA subset. It will be skipped as SKIPPED_NOT_OA.
- "NO_TGZ" / "SKIPPED_NO_TGZ":
  OA package not available for that PMCID.
- Date errors:
  CLI expects YYYY-MM-DD with correct month/day ranges.

"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import calendar
import functools
import argparse
import datetime as dt
import re

import requests
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry
from urllib.parse import quote_plus


# -----------------------------
# Constants
# -----------------------------
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PMC_OA_FCGI_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

DEFAULT_PAGE_SIZE = 1000
DEFAULT_TIMEOUT = 60

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
XML_EXTS = {".nxml", ".xml"}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# -----------------------------
# Small utils
# -----------------------------
def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def epmc_web_search_url(query: str) -> str:
    return f"https://europepmc.org/search?query={quote_plus(query)}"


def epmc_rest_search_url(query: str, page_size: int) -> str:
    return (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={quote_plus(query)}&format=json&pageSize={int(page_size)}"
    )


def pmc_article_url_from_pmcid(pmcid: str) -> str:
    pmcid = (pmcid or "").strip().upper()
    if not pmcid:
        return ""
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"


def europepmc_article_url_from_pmcid(pmcid: str) -> str:
    digits = "".join(ch for ch in (pmcid or "") if ch.isdigit())
    return f"https://europepmc.org/article/pmc/{digits}" if digits else ""


def make_session(user_agent: str, retries: int = 5, backoff: float = 0.6) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept": "*/*"})
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def validate_ymd_arg(value: str, argname: str) -> str:
    """
    Validate YYYY-MM-DD and give a helpful message like:
    'November 2025 has max 30 days; did you mean 2025-12-02?'
    """
    s = (value or "").strip()

    if not _DATE_RE.match(s):
        raise argparse.ArgumentTypeError(
            f"{argname}: expected YYYY-MM-DD, got {value!r}. Example: 2025-11-30"
        )

    y, m, d = map(int, s.split("-"))

    if not (1 <= m <= 12):
        raise argparse.ArgumentTypeError(
            f"{argname}: invalid month {m:02d} in {s!r}. Month must be 01..12."
        )

    max_day = calendar.monthrange(y, m)[1]

    # day must start from 01
    if d < 1:
        month_name = dt.date(y, m, 1).strftime("%B")
        raise argparse.ArgumentTypeError(
            f"{argname}: invalid day {d:02d} for {month_name} {y}. "
            f"Day must be 01..{max_day:02d}. Did you mean {y:04d}-{m:02d}-01?"
        )

    if d > max_day:
        month_name = dt.date(y, m, 1).strftime("%B")
        clamped = f"{y:04d}-{m:02d}-{max_day:02d}"
        raise argparse.ArgumentTypeError(
            f"{argname}: invalid day {d:02d} for {month_name} {y} (max {max_day:02d}). "
            f"Did you mean {clamped}?)"
        )

    return s

# -----------------------------
# DB schema + helpers
# -----------------------------
def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL;")

        con.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            pmcid TEXT PRIMARY KEY,
            seq INTEGER,
            source TEXT,
            pmid TEXT,
            doi TEXT,
            title TEXT,
            journal TEXT,
            pub_year INTEGER,
            pub_date TEXT,
            is_open_access INTEGER,
            raw_json TEXT,
            status TEXT,
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            updated_at TEXT
        );
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS figures (
            pmcid TEXT,
            figure_id TEXT,
            href TEXT,
            caption TEXT,
            file_path TEXT,
            sha256 TEXT,
            PRIMARY KEY (pmcid, href, file_path)
        );
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS licenses (
            pmcid TEXT PRIMARY KEY,
            license_type TEXT,
            license_url TEXT,
            license_text TEXT
        );
        """)

        con.execute("CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_articles_seq ON articles(seq);")
        con.commit()

        _ensure_seq_backfilled(con)
        con.commit()


def _ensure_seq_backfilled(con: sqlite3.Connection) -> None:
    # Backfill seq for existing DBs / rows with null seq.
    cols = {r[1] for r in con.execute("PRAGMA table_info(articles);").fetchall()}
    if "seq" not in cols:
        con.execute("ALTER TABLE articles ADD COLUMN seq INTEGER;")

    # If any rows have null seq, assign deterministically by pub_date then pmcid.
    missing = con.execute("SELECT COUNT(*) FROM articles WHERE seq IS NULL;").fetchone()[0]
    if missing and int(missing) > 0:
        rows = con.execute(
            "SELECT pmcid FROM articles ORDER BY COALESCE(pub_date,''), pmcid ASC;"
        ).fetchall()
        n = 0
        for i, (pmcid,) in enumerate(rows, start=1):
            con.execute("UPDATE articles SET seq=? WHERE pmcid=? AND seq IS NULL;", (i, pmcid))
            n += 1
        if n:
            con.commit()


def _next_seq(con: sqlite3.Connection) -> int:
    row = con.execute("SELECT COALESCE(MAX(seq), 0) FROM articles;").fetchone()
    return int(row[0] or 0) + 1


def upsert_article_with_seq(con: sqlite3.Connection, rec: Dict[str, Any], seq_if_new: int) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    con.execute("""
    INSERT INTO articles (
        pmcid, seq, source, pmid, doi, title, journal, pub_year, pub_date, is_open_access,
        raw_json, status, attempts, last_error, updated_at
    )
    VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
        COALESCE((SELECT status FROM articles WHERE pmcid=?), 'NEW'),
        COALESCE((SELECT attempts FROM articles WHERE pmcid=?), 0),
        COALESCE((SELECT last_error FROM articles WHERE pmcid=?), NULL),
        ?
    )
    ON CONFLICT(pmcid) DO UPDATE SET
        seq=COALESCE(articles.seq, excluded.seq),
        source=excluded.source,
        pmid=excluded.pmid,
        doi=excluded.doi,
        title=excluded.title,
        journal=excluded.journal,
        pub_year=excluded.pub_year,
        pub_date=excluded.pub_date,
        is_open_access=excluded.is_open_access,
        raw_json=excluded.raw_json,
        updated_at=excluded.updated_at;
    """, (
        rec["pmcid"], seq_if_new,
        rec.get("source"), rec.get("pmid"), rec.get("doi"),
        rec.get("title"), rec.get("journal"), rec.get("pub_year"), rec.get("pub_date"),
        int(bool(rec.get("is_open_access"))),
        json.dumps(rec, ensure_ascii=False),
        rec["pmcid"], rec["pmcid"], rec["pmcid"],
        now
    ))


def set_status(con: sqlite3.Connection, pmcid: str, status: str, err: Optional[str] = None) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    con.execute("""
    UPDATE articles
    SET status=?, last_error=?, updated_at=?
    WHERE pmcid=?;
    """, (status, err, now, pmcid))


def bump_attempts(con: sqlite3.Connection, pmcid: str) -> int:
    con.execute("UPDATE articles SET attempts = attempts + 1 WHERE pmcid=?;", (pmcid,))
    row = con.execute("SELECT attempts FROM articles WHERE pmcid=?;", (pmcid,)).fetchone()
    return int(row[0]) if row else 0


def iter_pending_articles(con: sqlite3.Connection, statuses: Tuple[str, ...]) -> List[Tuple[str, int]]:
    q = f"""
    SELECT pmcid, COALESCE(seq, 999999999) as seq
    FROM articles
    WHERE status IN ({','.join(['?'] * len(statuses))})
    ORDER BY seq ASC, COALESCE(pub_date,''), pmcid ASC;
    """
    rows = con.execute(q, statuses).fetchall()
    return [(r[0], int(r[1])) for r in rows]


def get_seq(con: sqlite3.Connection, pmcid: str) -> int:
    row = con.execute("SELECT COALESCE(seq, 0) FROM articles WHERE pmcid=?;", (pmcid,)).fetchone()
    return int(row[0] or 0)


# -----------------------------
# Discovery (Europe PMC)
# -----------------------------
@dataclass(frozen=True)
class DateRange:
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD

    def validate(self) -> None:
        dt.datetime.strptime(self.start, "%Y-%m-%d")
        dt.datetime.strptime(self.end, "%Y-%m-%d")

def epmc_src_clause(include_preprints: bool) -> str:
    # Biomedical-focused sources
    srcs = ["MED", "PMC"]
    if include_preprints:
        srcs.append("PPR")  # preprints
    return "(" + " OR ".join(f"SRC:{s}" for s in srcs) + ")"

def epmc_base_query(dr: DateRange, include_reviews: bool, include_preprints: bool = False) -> str:
    parts = [
        epmc_src_clause(include_preprints),
        "OPEN_ACCESS:y",
        f"FIRST_PDATE:[{dr.start} TO {dr.end}]",
        '(LICENSE:"CC0" OR LICENSE:"CC BY" OR LICENSE:"CC BY-SA")',
    ]

    if not include_reviews:
        parts.append('NOT (PUB_TYPE:review OR PUB_TYPE:"review-article" OR PUB_TYPE:"Review")')

    return " AND ".join(parts)


def epmc_topic_v1_clause() -> str:
    # Put ONLY the topic logic here (no dates/licenses), so it can be combined with base.
    q = r"""
    (
      (
        ("western blot" OR "western blotting" OR "western blot analysis"
         OR immunoblot* OR immunoblotting OR blotting
         OR "protein blot" OR "protein immunoblot*"
         OR "SDS-PAGE"
         OR "southern blot" OR "northern blot"
         OR "dot blot" OR "slot blot"
         OR "far-western blot" OR "southwestern blot"
         OR "eastern blot" OR "far-eastern blot")
      )
      OR
      (
        microscop* OR micrograph*
        OR (light AND microscop*) OR brightfield OR "bright field"
        OR "phase contrast" OR DIC OR "differential interference contrast"
        OR darkfield OR "dark field"
        OR "polarized light" OR polarised
        OR (fluorescen* AND microscop*) OR confocal OR "laser scanning confocal"
        OR epifluorescen* OR widefield OR "wide field"
        OR TIRF OR "two-photon" OR multiphoton
        OR "light sheet" OR lightsheet OR LSFM
        OR (electron AND microscop*) OR SEM OR TEM OR "cryo-EM" OR (cryo AND electron AND microscop*)
        OR histolog* OR histopatholog*
        OR immunofluorescen* OR immunohistochem* OR IHC
        OR "H&E" OR ("hematoxylin" AND eosin) OR ("haematoxylin" AND eosin)
        OR super-resol* OR STED OR STORM OR PALM OR SIM
      )
    )
    AND (fig* OR "fig." OR figure* OR panel* OR image* OR "scale bar" OR micrograph* OR "magnification" OR "µm" OR "inset")
    AND (biolog* OR biomed* OR medic* OR cell* OR tissu*)
    """
    return " ".join(q.split())


def build_epmc_query(
    dr: DateRange,
    mode: str,
    custom: Optional[str],
    include_reviews: bool,
    include_preprints: bool = False,
) -> str:
    base = epmc_base_query(dr, include_reviews=include_reviews, include_preprints=include_preprints)

    if mode == "base":
        return base

    if mode == "topic_v1":
        return f"({base}) AND ({epmc_topic_v1_clause()})"

    if mode == "custom_append":
        if not custom or not custom.strip():
            raise ValueError("custom_append requires --custom-query or --custom-query-file")
        return f"({base}) AND ({' '.join(custom.split())})"

    if mode == "custom_raw":
        # IMPORTANT: use exactly what user provided; do NOT auto-append SRC / review logic
        if not custom or not custom.strip():
            raise ValueError("custom_raw requires --custom-query or --custom-query-file")
        return " ".join(custom.split())

    raise ValueError(f"Unknown query mode: {mode}")


def fetch_europe_pmc_records(
    session: requests.Session,
    query: str,
    page_size: int,
    sleep_s: float,
) -> Iterable[Dict[str, Any]]:
    cursor = "*"
    while True:
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(page_size),
            "cursorMark": cursor,
            "resultType": "core",
        }
        r = session.get(EUROPE_PMC_SEARCH_URL, params=params, timeout=DEFAULT_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"Europe PMC search failed: HTTP {r.status_code} {r.text[:300]}")

        data = r.json()
        results = data.get("resultList", {}).get("result", []) or []
        for rec in results:
            yield rec

        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor or not results:
            break

        cursor = next_cursor
        if sleep_s > 0:
            time.sleep(sleep_s)


def normalize_epmc_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pmcid = rec.get("pmcid")
    if not pmcid:
        return None
    pmcid = pmcid.strip().upper()
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid

    pub_date = rec.get("firstPublicationDate") or rec.get("pubYear") or None
    pub_year = None
    try:
        if rec.get("pubYear"):
            pub_year = int(rec["pubYear"])
    except Exception:
        pub_year = None

    # Use isOpenAccess strictly (do NOT treat inEPMC as OA)
    is_oa = str(rec.get("isOpenAccess") or "").strip().lower() in ("y", "true", "1")

    return {
        "pmcid": pmcid,
        "source": rec.get("source"),
        "pmid": rec.get("pmid"),
        "doi": rec.get("doi"),
        "title": rec.get("title"),
        "journal": rec.get("journalTitle"),
        "pub_year": pub_year,
        "pub_date": pub_date,
        "is_open_access": is_oa,
        "epmc": rec,
    }


def write_discovery_report_near_db(
    db_path: Path,
    date_from: str,
    date_to: str,
    query: str,
    discovery_urls: List[str],
    inserted: int,
    started_monotonic: float,
    started_at_utc: str,
) -> Path:
    finished_at_utc = _utcnow_iso()
    elapsed_sec = round(time.monotonic() - started_monotonic, 3)

    payload = {
        "run": {
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "elapsed_sec": elapsed_sec,
        },
        "db_path": str(db_path.resolve()),
        "date_range": {"start": date_from, "end": date_to},
        "query": query,
        "discovery_urls": discovery_urls,
        "stats": {"inserted_upserted_with_pmcid": inserted},
    }

    out_path = db_path.parent / "pmc_oa_discovery_last_run.json"
    _atomic_write_json(out_path, payload)
    return out_path

def read_custom_query(args: argparse.Namespace) -> Optional[str]:
    if args.custom_query_file:
        return Path(args.custom_query_file).read_text(encoding="utf-8")
    return args.custom_query

def run_discovery(args: argparse.Namespace) -> None:
    db = Path(args.db)
    init_db(db)

    session = make_session(args.user_agent, retries=args.retries, backoff=args.backoff)

    started_monotonic = time.monotonic()
    started_at_utc = _utcnow_iso()

    dr = DateRange(args.start, args.end)
    if dr.start > dr.end:
        raise SystemExit(f"[ERROR] start date must be <= end date. Got start={dr.start} end={dr.end}")
    dr.validate()

    custom = read_custom_query(args)
    query = build_epmc_query(
        dr=dr,
        mode=args.query_mode,
        custom=custom,
        include_reviews=args.include_reviews,
        include_preprints=args.include_preprints,
    )

    print("\n" + "=" * 80)
    print("Generated Europe PMC query:")
    print("-" * 80)
    print(query)
    print("-" * 80)
    print("[optional] Test in browser (paste into Europe PMC search box): https://europepmc.org/")
    print("=" * 80 + "\n")

    # Write query to a TXT file (so you can copy/paste easily)
    query_txt = db.parent / "pmc_oa_discovery_query_last.txt"
    _atomic_write_text(query_txt, f"# mode={args.query_mode} include_reviews={args.include_reviews}\n{query}\n")
    print(f"[discovery] query saved: {query_txt}")

    discovery_urls = [
        epmc_web_search_url(query),
        epmc_rest_search_url(query, args.page_size),
    ]

    inserted = 0
    with sqlite3.connect(db) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        _ensure_seq_backfilled(con)
        next_seq = _next_seq(con)

        for rec in fetch_europe_pmc_records(session, query, args.page_size, args.sleep_epmc):
            norm = normalize_epmc_record(rec)
            if not norm:
                continue

            # if new pmcid, use next_seq; if existing, seq is preserved by upsert
            existing = con.execute("SELECT seq FROM articles WHERE pmcid=?;", (norm["pmcid"],)).fetchone()
            seq_for_insert = int(existing[0]) if (existing and existing[0] is not None) else next_seq
            if not (existing and existing[0] is not None):
                next_seq += 1

            upsert_article_with_seq(con, norm, seq_for_insert)
            inserted += 1

            if inserted % 500 == 0:
                con.commit()
                print(f"[discovery] upserted {inserted} PMCIDs...", flush=True)

        con.commit()

    report_path = write_discovery_report_near_db(
        db_path=db,
        date_from=args.start,
        date_to=args.end,
        query=query,
        discovery_urls=discovery_urls,
        inserted=inserted,
        started_monotonic=started_monotonic,
        started_at_utc=started_at_utc,
    )
    print(f"[discovery] done. Total upserted/updated with PMCID: {inserted}")
    print(f"[discovery] report saved: {report_path}")


# -----------------------------
# PMC OA service + extraction
# -----------------------------
def _oa_fcgi_xml(session: requests.Session, pmcid: str) -> str:
    r = session.get(PMC_OA_FCGI_URL, params={"id": pmcid}, timeout=DEFAULT_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"PMC OA fcgi failed for {pmcid}: HTTP {r.status_code} {r.text[:300]}")
    return r.text


def _oa_fcgi_error_code(xml_text: str) -> Optional[str]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None
    err = root.find(".//error")
    if err is not None:
        return err.attrib.get("code") or (err.text or "").strip() or "error"
    return None


def _extract_href_from_oa_xml(xml_text: str, fmt: str) -> Optional[str]:
    root = ET.fromstring(xml_text)
    for link in root.findall(".//link"):
        if link.attrib.get("format") == fmt and link.attrib.get("href"):
            return link.attrib["href"]
    return None


def _extract_license_from_oa_xml(xml_text: str) -> Optional[str]:
    root = ET.fromstring(xml_text)
    rec = root.find(".//record")
    if rec is not None:
        return rec.attrib.get("license")
    return None


def _ftp_to_https(url: str) -> str:
    # OA service commonly returns ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/... ; https works too.
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return "https://ftp.ncbi.nlm.nih.gov/" + url[len("ftp://ftp.ncbi.nlm.nih.gov/"):]
    return url


def download_pmc_oa_tgz(
    session: requests.Session,
    pmcid: str,
    out_dir: Path,
    sleep_s: float,
) -> Tuple[Optional[Path], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = out_dir / f"{pmcid}.tar.gz"

    xml_text = _oa_fcgi_xml(session, pmcid)

    # Handle explicit "not open access" error
    err_code = _oa_fcgi_error_code(xml_text)
    if err_code == "idIsNotOpenAccess":
        return None, xml_text

    # If file already exists, reuse
    if tgz_path.exists() and tgz_path.stat().st_size > 0:
        return tgz_path, xml_text

    tgz_href = _extract_href_from_oa_xml(xml_text, "tgz")
    if not tgz_href:
        return None, xml_text

    dl_url = _ftp_to_https(tgz_href)
    r = session.get(dl_url, timeout=DEFAULT_TIMEOUT, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"TGZ download failed for {pmcid}: HTTP {r.status_code} url={dl_url}")

    tmp = tgz_path.with_suffix(tgz_path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.replace(tmp, tgz_path)

    if tgz_path.stat().st_size < 10_000:
        raise RuntimeError(f"Downloaded tgz too small for {pmcid} url={dl_url} size={tgz_path.stat().st_size}")

    if sleep_s > 0:
        time.sleep(sleep_s)

    return tgz_path, xml_text


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_common_prefix(name: str, pmcid: str) -> str:
    parts = Path(name).parts
    if parts and parts[0].upper() == pmcid.upper():
        return str(Path(*parts[1:]))
    return name


def is_thumbnail_name(s: str) -> bool:
    s = s.lower()
    return ("thumb" in s) or ("thumbnail" in s) or s.endswith("_t.jpg") or s.endswith("_t.png")


def extract_only_xml(tgz_path: Path, extract_root: Path, pmcid: str) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    xml_path: Optional[Path] = None

    with tarfile.open(tgz_path, mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = _strip_common_prefix(m.name, pmcid)
            if not name:
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in XML_EXTS:
                continue

            out_path = extract_root / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not (out_path.exists() and out_path.stat().st_size > 0):
                f = tf.extractfile(m)
                if f:
                    tmp = out_path.with_suffix(out_path.suffix + ".part")
                    with open(tmp, "wb") as o:
                        shutil.copyfileobj(f, o)
                    os.replace(tmp, out_path)

            if xml_path is None or out_path.suffix.lower() == ".nxml":
                xml_path = out_path

    if not xml_path:
        raise RuntimeError("No XML found in tgz")
    return xml_path


def extract_selected_images(
    tgz_path: Path,
    extract_root: Path,
    pmcid: str,
    hrefs: List[str],
) -> Tuple[List[Path], Dict[str, Any]]:
    """
    Extract only images matching JATS hrefs (no custom caption filtering).
    Still skips obvious thumbnails.
    Returns (kept_paths, report_dict).
    """
    wanted: set[str] = set()
    wanted_basenames: set[str] = set()
    skipped_thumb = 0
    skipped_non_image = 0

    for h in hrefs:
        h = (h or "").strip().lstrip("./")
        if not h:
            continue
        wanted.add(h)
        wanted_basenames.add(Path(h).name)
        if Path(h).suffix == "":
            for ext in IMAGE_EXTS:
                wanted.add(h + ext)
                wanted_basenames.add(Path(h + ext).name)

    kept: List[Path] = []

    with tarfile.open(tgz_path, mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = _strip_common_prefix(m.name, pmcid)
            if not name:
                continue

            if name not in wanted and Path(name).name not in wanted_basenames:
                continue

            if is_thumbnail_name(name):
                skipped_thumb += 1
                continue

            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTS:
                skipped_non_image += 1
                continue

            out_path = extract_root / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not (out_path.exists() and out_path.stat().st_size > 0):
                f = tf.extractfile(m)
                if f:
                    tmp = out_path.with_suffix(out_path.suffix + ".part")
                    with open(tmp, "wb") as o:
                        shutil.copyfileobj(f, o)
                    os.replace(tmp, out_path)
            kept.append(out_path)

    report = {
        "requested_hrefs": len(hrefs),
        "wanted_entries": len(wanted),
        "kept_images": len(kept),
        "skipped_thumbnail_members": skipped_thumb,
        "skipped_non_image_members": skipped_non_image,
    }
    return kept, report


# -----------------------------
# XML parsing (license + figures)
# -----------------------------
def parse_license_from_jats(xml_path: Path) -> Dict[str, Optional[str]]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        return {"license_type": None, "license_url": None, "license_text": f"XML_PARSE_ERROR: {e}"}

    root = tree.getroot()
    ns = {"xlink": "http://www.w3.org/1999/xlink"}

    license_elems = root.findall(".//license")
    license_type = None
    license_url = None
    license_text = None

    for le in license_elems:
        license_type = le.attrib.get("license-type") or license_type
        ext = le.find(".//ext-link", ns)
        if ext is not None:
            href = ext.attrib.get(f"{{{ns['xlink']}}}href")
            if href:
                license_url = href
        txt = " ".join("".join(le.itertext()).split()).strip()
        if txt:
            license_text = txt
        if license_text or license_url or license_type:
            break

    if not (license_text or license_url or license_type):
        perms = root.find(".//permissions")
        if perms is not None:
            license_text = " ".join("".join(perms.itertext()).split()).strip() or None

    return {"license_type": license_type, "license_url": license_url, "license_text": license_text}


def parse_figures_from_jats(xml_path: Path) -> List[Dict[str, Optional[str]]]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    ns = {"xlink": "http://www.w3.org/1999/xlink"}
    out: List[Dict[str, Optional[str]]] = []

    for fig in root.findall(".//fig"):
        fig_id = fig.attrib.get("id")

        cap = fig.find(".//caption")
        caption_text = None
        if cap is not None:
            caption_text = " ".join("".join(cap.itertext()).split()).strip() or None

        for g in fig.findall(".//graphic"):
            href = g.attrib.get(f"{{{ns['xlink']}}}href")
            if href:
                out.append({"figure_id": fig_id, "href": href, "caption": caption_text})

        for g in fig.findall(".//inline-graphic"):
            href = g.attrib.get(f"{{{ns['xlink']}}}href")
            if href:
                out.append({"figure_id": fig_id, "href": href, "caption": caption_text})

    return out


def match_href_to_file(extract_root: Path, href: str) -> Optional[Path]:
    href = (href or "").strip().lstrip("./")
    if not href:
        return None

    p = extract_root / href
    if p.exists():
        return p

    # Try any known extension if href omits it
    if Path(href).suffix == "":
        for ext in IMAGE_EXTS:
            cand = extract_root / f"{href}{ext}"
            if cand.exists():
                return cand

    # Fallback: brute search by basename
    base = Path(href).name
    candidates = list(extract_root.rglob(base))
    if candidates:
        return candidates[0]

    if Path(href).suffix == "":
        for ext in IMAGE_EXTS:
            candidates = list(extract_root.rglob(base + ext))
            if candidates:
                return candidates[0]

    return None


# -----------------------------
# Per-paper manifest + metadata
# -----------------------------
def write_metadata_json(path: Path, data: Dict[str, Any]) -> None:
    _atomic_write_json(path, data)


def build_and_save_paper_metadata(
    con: sqlite3.Connection,
    pmcid: str,
    extract_root: Path,
    oa_license: Optional[str],
) -> Dict[str, Any]:
    row = con.execute(
        "SELECT raw_json, pmid, doi, title, journal, pub_date, source FROM articles WHERE pmcid=?",
        (pmcid,),
    ).fetchone()
    raw_json = json.loads(row[0]) if row and row[0] else {}

    lic_row = con.execute(
        "SELECT license_type, license_url, license_text FROM licenses WHERE pmcid=?",
        (pmcid,),
    ).fetchone()

    fig_rows = con.execute(
        """SELECT figure_id, href, caption, file_path, sha256
           FROM figures WHERE pmcid=?
           ORDER BY figure_id, href;""",
        (pmcid,),
    ).fetchall()

    fig_map: Dict[str, Any] = {}
    for fig_id, href, caption, file_path, sha in fig_rows:
        key = fig_id or href or "unknown"
        ent = fig_map.setdefault(key, {"figure_id": fig_id, "href": href, "caption": caption, "images": []})
        ent["images"].append({"file_path": file_path, "sha256": sha})

    meta: Dict[str, Any] = {
        "pmcid": pmcid,
        "pmid": row[1] if row else None,
        "doi": row[2] if row else None,
        "title": row[3] if row else None,
        "journal": row[4] if row else None,
        "pub_date": row[5] if row else None,
        "source": row[6] if row else None,
        "urls": {
            "pmc": pmc_article_url_from_pmcid(pmcid),
            "europe_pmc": europepmc_article_url_from_pmcid(pmcid),
        },
        "license": {
            "oa_fcgi_license": oa_license,
            "jats_license_type": lic_row[0] if lic_row else None,
            "license_url": lic_row[1] if lic_row else None,
            "license_text": lic_row[2] if lic_row else None,
        },
        "figures": list(fig_map.values()),
        "raw": raw_json,
    }

    write_metadata_json(extract_root / "metadata.json", meta)
    return meta


def build_and_save_figures_manifest(
    pmcid: str,
    extract_root: Path,
    figures: List[Dict[str, Optional[str]]],
) -> Dict[str, Any]:
    # For inspection: what existed in XML vs what we actually have on disk
    resolved = []
    missing = 0
    skipped_thumbnail = 0

    resolved_paths: set[str] = set()

    for f in figures:
        href = (f.get("href") or "").strip()
        ent = {"figure_id": f.get("figure_id"), "href": href, "caption": f.get("caption")}
        if not href:
            ent["status"] = "no_href"
        elif is_thumbnail_name(href):
            ent["status"] = "skipped_thumbnail"
            skipped_thumbnail += 1
        else:
            p = match_href_to_file(extract_root, href)
            if p and p.exists():
                rel = str(p.relative_to(extract_root))
                ent["status"] = "extracted"
                ent["resolved_path"] = rel
                resolved_paths.add(rel)
            else:
                ent["status"] = "missing_in_package"
                missing += 1
        resolved.append(ent)

    # Also list image files present but not referenced by <fig>
    unref = []
    for p in extract_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rel = str(p.relative_to(extract_root))
            if rel not in resolved_paths:
                unref.append(rel)

    payload = {
        "pmcid": pmcid,
        "fig_entries_in_xml": len(figures),
        "missing_in_package": missing,
        "skipped_thumbnail_hrefs": skipped_thumbnail,
        "resolved": resolved,
        "unreferenced_images": sorted(unref),
    }

    _atomic_write_json(extract_root / "figures_manifest.json", payload)
    return payload


# -----------------------------
# Layout (flat vs batched)
# -----------------------------
def paper_dir_for(
    out_root: Path,
    paper_root: str,
    layout: str,
    batch_size: int,
    seq: int,
    pmcid: str,
) -> Path:
    base = out_root / paper_root
    if layout == "flat":
        return base / pmcid

    # default batched
    if batch_size <= 0:
        batch_size = 100
    batch_id = (max(seq, 1) - 1) // batch_size
    return base / f"batch_{batch_id:05d}" / pmcid


# -----------------------------
# Run pipeline
# -----------------------------
def _pick_stats(stats: dict, mode: str) -> dict:
    if mode == "none":
        return {}
    if mode == "basic":
        keys = ["done", "failed", "skip_not_oa", "skip_no_tgz"]
        return {k: stats.get(k, 0) for k in keys}
    return dict(stats)


def write_run_report(out_root: Path, payload: Dict[str, Any]) -> Path:
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "pmc_oa_run_last.json"
    _atomic_write_json(out_path, payload)
    if not out_path.exists():
        raise RuntimeError(f"Run report not written to disk: {out_path}")
    return out_path


def resolve_out_dir(db_path: str, out_arg: str | None) -> Path:
    db_p = Path(db_path)
    out_p = Path(out_arg) if out_arg else (db_p.parent / "data")
    out_p.mkdir(parents=True, exist_ok=True)
    return out_p


def run_download_extract(args: argparse.Namespace) -> None:
    out_dir = resolve_out_dir(args.db, args.out)
    print(f"Using output directory: {out_dir.resolve()}")
    args.out = str(out_dir)

    db = Path(args.db)
    out_root = Path(args.out)
    pkg_dir = out_root / "raw_packages"

    init_db(db)

    # ensure visible dirs
    out_root.mkdir(parents=True, exist_ok=True)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    session = make_session(args.user_agent, retries=args.retries, backoff=args.backoff)

    started_monotonic = time.monotonic()
    started_at_utc = _utcnow_iso()

    stats = {
        "done": 0,
        "done_no_images": 0,
        "failed": 0,
        "skip_not_oa": 0,
        "skip_no_tgz": 0,
        "skip_license": 0,
        "max_attempts": 0,
    }

    with sqlite3.connect(db) as con:
        con.execute("PRAGMA journal_mode=WAL;")

        # include in-progress statuses so resume works after crash
        pending = iter_pending_articles(con, ("NEW", "FAILED", "DOWNLOADING", "EXTRACTING"))
        if args.limit and args.limit > 0:
            pending = pending[: args.limit]

        total = len(pending)
        print(f"[work] pending PMCIDs: {total}")

        with tqdm(total=total, desc="download+extract", unit="paper", dynamic_ncols=True) as pbar:
            for idx, (pmcid, seq) in enumerate(pending, start=1):
                try:
                    attempts = bump_attempts(con, pmcid)
                    if args.max_attempts and attempts > args.max_attempts:
                        set_status(con, pmcid, "FAILED", err=f"max_attempts_exceeded({attempts})")
                        con.commit()
                        stats["max_attempts"] += 1
                        stats["failed"] += 1
                        continue

                    set_status(con, pmcid, "DOWNLOADING")
                    con.commit()

                    tgz, oa_xml = download_pmc_oa_tgz(session, pmcid, pkg_dir, args.sleep_pmc)

                    # OA license (if present)
                    oa_license = None
                    try:
                        oa_license = _extract_license_from_oa_xml(oa_xml)
                    except Exception:
                        oa_license = None

                    # oa.fcgi explicit "not OA" → skip
                    if tgz is None:
                        err_code = _oa_fcgi_error_code(oa_xml)
                        if err_code == "idIsNotOpenAccess":
                            set_status(con, pmcid, "SKIPPED_NOT_OA", err="oa_fcgi_idIsNotOpenAccess")
                            con.commit()
                            stats["skip_not_oa"] += 1
                            continue
                        else:
                            set_status(con, pmcid, "SKIPPED_NO_TGZ", err=f"oa_fcgi_no_tgz_link_or_error:{err_code}")
                            con.commit()
                            stats["skip_no_tgz"] += 1
                            continue

                    # Save OA license early
                    con.execute("""
                    INSERT INTO licenses(pmcid, license_type, license_url, license_text)
                    VALUES(?, ?, NULL, NULL)
                    ON CONFLICT(pmcid) DO UPDATE SET license_type=excluded.license_type;
                    """, (pmcid, oa_license))

                    set_status(con, pmcid, "EXTRACTING")
                    con.commit()

                    # Per-paper folder path based on layout
                    paper_dir = paper_dir_for(
                        out_root=out_root,
                        paper_root=args.paper_root,
                        layout=args.layout,
                        batch_size=args.batch_size,
                        seq=seq,
                        pmcid=pmcid,
                    )
                    paper_dir.mkdir(parents=True, exist_ok=True)

                    # 1) extract XML
                    xml_path = extract_only_xml(tgz, paper_dir, pmcid)

                    # 2) parse license from JATS and enrich
                    lic = parse_license_from_jats(xml_path)
                    con.execute("""
                    INSERT INTO licenses(pmcid, license_type, license_url, license_text)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(pmcid) DO UPDATE SET
                      license_type=COALESCE(excluded.license_type, licenses.license_type),
                      license_url=excluded.license_url,
                      license_text=excluded.license_text;
                    """, (pmcid, lic.get("license_type") or oa_license, lic.get("license_url"), lic.get("license_text")))

                    # 3) parse ALL figures (no custom filtering)
                    figs = parse_figures_from_jats(xml_path)
                    hrefs = [f["href"] for f in figs if f.get("href")]

                    # 4) extract only images referenced by those hrefs
                    kept_images, extract_report = extract_selected_images(tgz, paper_dir, pmcid, hrefs)

                    # 5) insert figure rows for resolved files
                    saved = 0
                    for f in figs:
                        href = f.get("href") or ""
                        if not href or is_thumbnail_name(href):
                            continue
                        matched = match_href_to_file(paper_dir, href)
                        if matched and matched.exists():
                            digest = sha256_file(matched)
                            con.execute("""
                            INSERT OR IGNORE INTO figures(pmcid, figure_id, href, caption, file_path, sha256)
                            VALUES(?, ?, ?, ?, ?, ?);
                            """, (
                                pmcid,
                                f.get("figure_id"),
                                href,
                                f.get("caption"),
                                str(matched.relative_to(out_root)),
                                digest
                            ))
                            saved += 1

                    # 6) per-paper inspection files
                    build_and_save_paper_metadata(con=con, pmcid=pmcid, extract_root=paper_dir, oa_license=oa_license)
                    build_and_save_figures_manifest(pmcid=pmcid, extract_root=paper_dir, figures=figs)

                    # 7) final status
                    if saved > 0:
                        set_status(con, pmcid, "DONE", err=None)
                        stats["done"] += 1
                    else:
                        set_status(con, pmcid, "DONE_NO_IMAGES", err=f"no_images_saved; report={extract_report}")
                        stats["done_no_images"] += 1

                    con.commit()

                    if args.delete_raw_packages and tgz is not None:
                        try:
                            tgz.unlink(missing_ok=True)
                        except Exception as e:
                            tqdm.write(f"[WARN] Could not delete package {tgz}: {e}")

                except Exception as e:
                    set_status(con, pmcid, "FAILED", err=str(e))
                    con.commit()
                    stats["failed"] += 1
                    tqdm.write(f"[ERROR] {pmcid}: {e}")
                    traceback.print_exc(file=sys.stderr)

                finally:
                    pbar.update(1)
                    mode = getattr(args, "pbar_stats", "basic")
                    if mode != "none" and (idx % 5 == 0 or idx == total):
                        pbar.set_postfix(_pick_stats(stats, mode), refresh=False)

    finished_at_utc = _utcnow_iso()
    elapsed_sec = round(time.monotonic() - started_monotonic, 3)

    report_payload = {
        "run": {
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "elapsed_sec": elapsed_sec,
        },
        "db_path": str(db.resolve()),
        "out_root": str(out_root.resolve()),
        "args": {
            "limit": args.limit,
            "max_attempts": args.max_attempts,
            "sleep_pmc": args.sleep_pmc,
            "layout": args.layout,
            "batch_size": args.batch_size,
            "paper_root": args.paper_root,
            "pbar_stats": getattr(args, "pbar_stats", "basic"),
            "delete_raw_packages": bool(args.delete_raw_packages),
        },
        "pending_total": int(total),
        "stats": stats,
    }

    report_path = write_run_report(out_root, report_payload)
    print("[work] finished.")
    print(f"[work] report saved: {report_path}")


# -----------------------------
# Export
# -----------------------------
def export_reports(args: argparse.Namespace) -> None:
    db = Path(args.db)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row

        lic_rows = con.execute("""
        SELECT a.pmcid, a.pub_date, a.title, a.journal, a.doi, a.pmid,
               l.license_type, l.license_url
        FROM articles a
        LEFT JOIN licenses l ON a.pmcid = l.pmcid
        WHERE a.status IN ('DONE','DONE_NO_IMAGES')
        ORDER BY COALESCE(a.seq, 999999999), COALESCE(a.pub_date,''), a.pmcid ASC;
        """).fetchall()

        lic_csv = out_root / "licenses.csv"
        with open(lic_csv, "w", encoding="utf-8") as f:
            f.write("pmcid,pub_date,title,journal,doi,pmid,license_type,license_url\n")

            def esc(x: Any) -> str:
                if x is None:
                    return ""
                s = str(x).replace('"', '""')
                return f'"{s}"' if ("," in s or "\n" in s or '"' in s) else s

            for r in lic_rows:
                f.write(",".join([
                    esc(r["pmcid"]), esc(r["pub_date"]), esc(r["title"]), esc(r["journal"]),
                    esc(r["doi"]), esc(r["pmid"]), esc(r["license_type"]), esc(r["license_url"])
                ]) + "\n")

        fig_rows = con.execute("""
        SELECT pmcid, figure_id, href, caption, file_path, sha256
        FROM figures
        ORDER BY pmcid ASC, figure_id ASC;
        """).fetchall()

        fig_csv = out_root / "figures.csv"
        with open(fig_csv, "w", encoding="utf-8") as f:
            f.write("pmcid,figure_id,href,caption,file_path,sha256\n")

            def esc(x: Any) -> str:
                if x is None:
                    return ""
                s = str(x).replace('"', '""')
                return f'"{s}"' if ("," in s or "\n" in s or '"' in s) else s

            for r in fig_rows:
                f.write(",".join([
                    esc(r["pmcid"]), esc(r["figure_id"]), esc(r["href"]),
                    esc(r["caption"]), esc(r["file_path"]), esc(r["sha256"])
                ]) + "\n")

    print(f"[export] wrote: {lic_csv}")
    print(f"[export] wrote: {fig_csv}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PMC OA dataset builder (Raw Packages + Figures) — Resumable")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="Discover OA PMCIDs in a date range and store in SQLite")
    d.add_argument("--start", required=True, type=functools.partial(validate_ymd_arg, argname="--start"),
                help="Start date YYYY-MM-DD (e.g., 2025-11-30)")
    d.add_argument("--end", required=True, type=functools.partial(validate_ymd_arg, argname="--end"),
                help="End date YYYY-MM-DD (e.g., 2025-12-02)")
    d.add_argument("--db", default="out_pmc/db.sqlite", help="SQLite DB path")
    d.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    d.add_argument("--sleep-epmc", type=float, default=0.1)
    d.add_argument("--user-agent", default="pmc-oa-dataset-builder/1.0 (contact: you@example.com)")
    d.add_argument("--retries", type=int, default=5)
    d.add_argument("--backoff", type=float, default=0.6)
    d.add_argument(
        "--query-mode",
        choices=["base", "topic_v1", "custom_append", "custom_raw"],
        default="topic_v1",
        help="Query profile: base(no topic keywords), topic_v1(your microscopy/blot query), custom_append(base AND custom), custom_raw(use custom as full query).",
    )
    d.add_argument("--custom-query", default=None, help="Custom query string (used by custom_* modes).")
    d.add_argument("--custom-query-file", default=None, help="Path to a text file containing the custom query (recommended).")
    d.add_argument(
        "--include-reviews",
        action="store_true",
        help="Include review articles. Default behavior excludes reviews.",
    )
    d.add_argument(
        "--include-preprints",
        action="store_true",
        help="Include preprints by adding SRC:PPR (default: off)",
    )


    w = sub.add_parser("run", help="Download + extract packages for discovered PMCIDs (resumable)")
    w.add_argument("--db", required=True, help="SQLite DB path")
    w.add_argument("--out", default=None, help='Output root dir. If not provided, uses "<db_parent>/data".')
    w.add_argument("--limit", type=int, default=0)
    w.add_argument("--max-attempts", type=int, default=6)
    w.add_argument("--sleep-pmc", type=float, default=0.15)
    w.add_argument("--user-agent", default="pmc-oa-dataset-builder/1.0 (contact: you@example.com)")
    w.add_argument("--retries", type=int, default=6)
    w.add_argument("--backoff", type=float, default=0.7)
    w.add_argument(
        "--delete-raw-packages",
        action="store_true",
        help="Delete downloaded .tar.gz package after successful extraction (saves disk).",
    )

    # layout options
    w.add_argument("--layout", choices=["flat", "batched"], default="batched",
                   help="flat: papers/PMCxxxx ; batched: papers/batch_xxxxx/PMCxxxx (default).")
    w.add_argument("--batch-size", type=int, default=100,
                   help="How many paper folders per batch folder when layout=batched (default 100).")
    w.add_argument("--paper-root", default="papers",
                   help='Root folder under out/ for per-paper folders (default "papers").')

    w.add_argument("--pbar-stats", choices=["basic", "all", "none"], default="basic")

    e = sub.add_parser("export", help="Export licenses.csv and figures.csv from SQLite")
    e.add_argument("--db", default="out_pmc/db.sqlite")
    e.add_argument("--out", default="out_pmc")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "discover":
        run_discovery(args)
    elif args.cmd == "run":
        run_download_extract(args)
    elif args.cmd == "export":
        export_reports(args)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()