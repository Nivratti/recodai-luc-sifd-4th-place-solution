#!/usr/bin/env python3
"""
Build an A4 print-ready PDF from images in a folder filtered by filename keyword.

Features:
- 1 image per page
- Title line on each page: "N. filename"
- DEFAULT page-mode=auto: landscape pages used for wide images (best fill)
- OPTIONAL page-mode=portrait: all pages portrait; rotate wide images to fit (toggle)
- Keeps aspect ratio, fits to page with margins
- Writes stats (JSON + CSV) into output directory alongside PDF
- No "Generated ..." footer text
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# --- progress (optional) ---
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

def pbar(iterable, desc: str, enabled: bool = True):
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, unit="file", dynamic_ncols=True)
    return iterable

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class FileInfo:
    index: int
    path: str
    name: str
    bytes: int
    orig_width: int
    orig_height: int
    final_width: int
    final_height: int
    mode: str
    page_orientation: str          # "portrait" or "landscape"
    rotated_degrees: int           # 0 or 90
    draw_w_pt: float               # drawn width in PDF points
    draw_h_pt: float               # drawn height in PDF points
    scale: float


def iter_files(root: Path, recursive: bool) -> Iterable[Path]:
    yield from (root.rglob("*") if recursive else root.glob("*"))


def open_image_for_pdf(p: Path) -> Image.Image:
    im = Image.open(p)
    # Respect EXIF orientation
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    # Convert to RGB to avoid alpha/palette edge cases in PDFs
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if im.mode == "L":
        im = im.convert("RGB")
    return im


def fit_box(img_w: int, img_h: int, box_w: float, box_h: float) -> Tuple[float, float, float]:
    scale = min(box_w / img_w, box_h / img_h)
    return img_w * scale, img_h * scale, scale


def build_pdf(
    files: List[Path],
    out_pdf: Path,
    page_mode: str,                       # "auto" or "portrait"
    portrait_rotate_to_fit: bool,         # used only when page_mode="portrait"
    margin_inch: float = 0.5,
    title_font: str = "Helvetica",
    title_size: int = 12,
    title_gap_inch: float = 0.25,
    show_progress: bool = True,
) -> List[FileInfo]:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    margin = margin_inch * inch
    title_gap = title_gap_inch * inch
    c = canvas.Canvas(str(out_pdf))

    infos: List[FileInfo] = []

    for idx, p in enumerate(pbar(files, "Rendering PDF pages", enabled=show_progress), start=1):
        im = open_image_for_pdf(p)
        orig_w, orig_h = im.size
        rotated_degrees = 0

        # Decide page size/orientation
        if page_mode == "auto":
            # Same behavior as your previous PDF: wide -> landscape page; tall -> portrait page
            if orig_w > orig_h:
                page_w, page_h = landscape(A4)
                page_orient = "landscape"
            else:
                page_w, page_h = A4
                page_orient = "portrait"
        elif page_mode == "portrait":
            page_w, page_h = A4
            page_orient = "portrait"
        else:
            raise ValueError(f"Unknown page_mode: {page_mode}")

        c.setPageSize((page_w, page_h))

        # Title (filename only)
        c.setFont(title_font, title_size)
        title_y = page_h - margin
        c.drawString(margin, title_y, f"{idx}. {p.name}")

        # Box for image (leave room for title)
        top_reserved = margin + title_gap + title_size * 1.2
        box_x = margin
        box_y = margin
        box_w = page_w - 2 * margin
        box_h = page_h - top_reserved - box_y

        # Optional rotate-to-fit in portrait mode (rotate the IMAGE, keep page portrait)
        if page_mode == "portrait" and portrait_rotate_to_fit:
            w0, h0, s0 = fit_box(orig_w, orig_h, box_w, box_h)     # original
            w1, h1, s1 = fit_box(orig_h, orig_w, box_w, box_h)     # if rotated 90°
            if s1 > s0:
                im = im.rotate(90, expand=True)
                rotated_degrees = 90

        final_w, final_h = im.size

        draw_w, draw_h, scale = fit_box(final_w, final_h, box_w, box_h)
        x = box_x + (box_w - draw_w) / 2
        y = box_y + (box_h - draw_h) / 2

        c.drawImage(
            ImageReader(im),
            x, y,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            mask="auto",
        )
        c.showPage()

        infos.append(
            FileInfo(
                index=idx,
                path=str(p.resolve()),
                name=p.name,
                bytes=p.stat().st_size,
                orig_width=orig_w,
                orig_height=orig_h,
                final_width=final_w,
                final_height=final_h,
                mode=im.mode,
                page_orientation=page_orient,
                rotated_degrees=rotated_degrees,
                draw_w_pt=float(draw_w),
                draw_h_pt=float(draw_h),
                scale=float(scale),
            )
        )

    c.save()
    return infos


def write_stats(out_dir: Path, infos: List[FileInfo], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    (out_dir / "pdf_build_stats.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # CSV
    csv_path = out_dir / "pdf_build_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(infos[0]).keys()) if infos else [
            "index","name","path","bytes","orig_width","orig_height","final_width","final_height",
            "mode","page_orientation","rotated_degrees","draw_w_pt","draw_h_pt","scale"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for x in infos:
            w.writerow(asdict(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True, help="Folder containing images")
    ap.add_argument("--keyword", type=str, default="overlay_boundaries",
                    help="Filename must contain this keyword (case-insensitive)")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders too")
    ap.add_argument("--exts", type=str, default="png,jpg,jpeg,tif,tiff,bmp,webp",
                    help="Comma-separated extensions to include")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for PDF + stats")
    ap.add_argument("--pdf-name", type=str, default=None, help="Output PDF filename (default auto)")

    # Toggle between old behavior and new behavior
    ap.add_argument("--page-mode", choices=["auto", "portrait"], default="auto",
                    help="auto=wide images get landscape pages (DEFAULT). portrait=all pages portrait A4.")
    ap.add_argument("--portrait-rotate-to-fit", action=argparse.BooleanOptionalAction, default=True,
                    help="Only used when --page-mode portrait. If true, rotate wide images 90° when it increases fit.")

    ap.add_argument("--margin-inch", type=float, default=0.5, help="Page margin in inches")
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                help="Show tqdm progress bars (default: true)")

    args = ap.parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"--images-dir not found: {args.images_dir}")

    exts = {"." + e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()}
    exts = exts or SUPPORTED_EXTS
    keyword = args.keyword.lower()

    # Collect matches
    candidates: List[Path] = []
    for p in pbar(iter_files(args.images_dir, args.recursive), "Scanning files", enabled=args.progress):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if keyword not in p.name.lower():
            continue
        candidates.append(p)

    candidates.sort(key=lambda x: natural_key(x.name))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = args.pdf_name or f"print_{args.keyword}.pdf"
    pdf_path = args.out_dir / pdf_name

    if not candidates:
        print(f"[!] No files found in {args.images_dir} matching keyword='{args.keyword}' and exts={sorted(exts)}")
        summary = {
            "pdf": str(pdf_path.resolve()),
            "pdf_bytes": None,
            "num_images": 0,
            "images_dir": str(args.images_dir.resolve()),
            "keyword": args.keyword,
            "recursive": args.recursive,
            "exts": sorted(exts),
            "page_mode": args.page_mode,
            "portrait_rotate_to_fit": args.portrait_rotate_to_fit,
            "margin_inch": args.margin-inch if False else args.margin_inch,  # keep python happy
            "images": [],
            "note": "No files matched filter; PDF not created",
        }
        # write empty stats
        (args.out_dir / "pdf_build_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (args.out_dir / "pdf_build_stats.csv").write_text(
            "index,name,path,bytes,orig_width,orig_height,final_width,final_height,mode,page_orientation,rotated_degrees,draw_w_pt,draw_h_pt,scale\n",
            encoding="utf-8",
        )
        return

    infos = build_pdf(
        files=candidates,
        out_pdf=pdf_path,
        page_mode=args.page_mode,
        portrait_rotate_to_fit=args.portrait_rotate_to_fit,
        margin_inch=args.margin_inch,
        show_progress=args.progress
    )

    summary = {
        "pdf": str(pdf_path.resolve()),
        "pdf_bytes": pdf_path.stat().st_size,
        "num_images": len(infos),
        "images_dir": str(args.images_dir.resolve()),
        "keyword": args.keyword,
        "recursive": args.recursive,
        "exts": sorted(exts),
        "page_mode": args.page_mode,
        "portrait_rotate_to_fit": args.portrait_rotate_to_fit,
        "margin_inch": args.margin_inch,
        "images": [asdict(x) for x in infos],
    }

    write_stats(args.out_dir, infos, summary)

    # Console stats
    print("\n=== PDF BUILD SUMMARY ===")
    print(f"Images dir : {args.images_dir.resolve()}")
    print(f"Keyword    : {args.keyword}")
    print(f"Recursive  : {args.recursive}")
    print(f"Page mode  : {args.page_mode} (portrait_rotate_to_fit={args.portrait_rotate_to_fit})")
    print(f"Found      : {len(infos)} file(s)")
    print(f"Output PDF : {pdf_path.resolve()} ({pdf_path.stat().st_size:,} bytes)")
    print(f"Stats JSON : {(args.out_dir / 'pdf_build_stats.json').resolve()}")
    print(f"Stats CSV  : {(args.out_dir / 'pdf_build_stats.csv').resolve()}")

    print("\nFiles:")
    for x in infos:
        rot = f"rot{ x.rotated_degrees }" if x.rotated_degrees else "rot0"
        print(f"  {x.index:03d}. {x.name}  [{x.orig_width}x{x.orig_height}] -> [{x.final_width}x{x.final_height}] "
              f"{x.page_orientation} {rot}  {x.bytes:,} bytes")


if __name__ == "__main__":
    main()
