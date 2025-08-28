from __future__ import annotations
import os, json, shutil
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

from flask import (
    Flask, request, render_template, redirect, url_for,
    send_file, abort, flash, jsonify
)
from slugify import slugify
from pypdf import PdfReader, PdfWriter
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path

APP_NAME = "libreturner"
BASE_DIR = Path(os.environ.get("LIBRETURNER_HOME", Path.home() / ".libreturner"))
PIECES_DIR = BASE_DIR / "pieces"
PROGRAMMES_DIR = BASE_DIR / "programmes"

# Instantiate Flask app BEFORE any route decorators are used
app = Flask(__name__)
app.secret_key = "dev-secret"  # replace in production

def ensure_dirs():
    for d in [BASE_DIR, PIECES_DIR, PROGRAMMES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def path_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0

def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: Path, data: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_page_ranges(s: Optional[str], max_pages: Optional[int]=None) -> List[int]:
    if not s or not s.strip():
        return list(range(1, (max_pages or 0) + 1)) if max_pages else []
    pages = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if start <= end:
                rng = range(start, end + 1)
            else:
                rng = range(end, start + 1)
            pages.update(rng)
        else:
            pages.add(int(part))
    out = [p for p in sorted(pages) if p >= 1 and (max_pages is None or p <= max_pages)]
    return out

def pages_to_range_str(pages: List[int]) -> str:
    if not pages:
        return ""
    pages = sorted(set(int(p) for p in pages))
    ranges = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = p
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)

def get_pdf_total_pages(pdf_path: Path) -> int:
    info = pdfinfo_from_path(str(pdf_path))
    return int(info.get("Pages", 0))

def get_selected_pages_pdf(cfg: dict, total_pages: int) -> List[int]:
    sel = cfg.get("selected_pages")
    if isinstance(sel, list) and len(sel) > 0:
        return [p for p in sel if 1 <= p <= total_pages]
    pr = parse_page_ranges(cfg.get("page_ranges"), max_pages=total_pages)
    if pr:
        return pr
    return list(range(1, total_pages + 1))

def newest_source_mtime_for_piece(piece_dir: Path, cfg: dict) -> float:
    times = [path_mtime(piece_dir / "config.json")]
    if cfg["type"] == "pdf":
        times.append(path_mtime(piece_dir / cfg["source_pdf"]))
    else:
        for img in cfg.get("images", []):
            times.append(path_mtime(piece_dir / img["path"]))
    return max(times) if times else 0.0

def needs_rebuild_piece(piece_dir: Path, cfg: dict) -> bool:
    out_pdf = piece_dir / "outputs" / f"{cfg['slug']}.pdf"
    if not out_pdf.exists():
        return True
    src_mtime = newest_source_mtime_for_piece(piece_dir, cfg)
    return path_mtime(out_pdf) < src_mtime

def build_piece_pdf(piece_dir: Path, cfg: dict) -> Path:
    outputs = piece_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    out_pdf = outputs / f"{cfg['slug']}.pdf"
    writer = PdfWriter()

    if cfg["type"] == "pdf":
        pdf_path = piece_dir / cfg["source_pdf"]
        reader = PdfReader(str(pdf_path))
        pages = get_selected_pages_pdf(cfg, len(reader.pages))
        for p1 in pages:
            writer.add_page(reader.pages[p1 - 1])

    elif cfg["type"] == "images":
        imgs = [img for img in cfg.get("images", []) if img.get("enabled", True)]
        for img in imgs:
            img_path = piece_dir / img["path"]
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")
            im = Image.open(img_path)
            rotate = int(img.get("rotate", 0))
            if rotate % 360 != 0:
                im = im.rotate(-rotate, expand=True)
            im = im.convert("RGB")
            buf = BytesIO()
            im.save(buf, format="PDF", resolution=300.0)
            buf.seek(0)
            r = PdfReader(buf)
            writer.add_page(r.pages[0])
    else:
        raise ValueError("Unknown piece type")

    with out_pdf.open("wb") as f:
        writer.write(f)

    cfg["last_built"] = now_iso()
    save_json(piece_dir / "config.json", cfg)
    return out_pdf

def ensure_piece_built(slug: str) -> Path:
    piece_dir = PIECES_DIR / slug
    cfg = load_json(piece_dir / "config.json")
    if needs_rebuild_piece(piece_dir, cfg):
        return build_piece_pdf(piece_dir, cfg)
    return piece_dir / "outputs" / f"{cfg['slug']}.pdf"

def needs_rebuild_programme(prog_dir: Path, cfg: dict) -> bool:
    out_pdf = prog_dir / "outputs" / f"{cfg['slug']}.pdf"
    if not out_pdf.exists():
        return True
    out_mtime = path_mtime(out_pdf)
    times = [path_mtime(prog_dir / "config.json")]
    for pslug in cfg.get("pieces", []):
        pdir = PIECES_DIR / pslug
        pcfg = load_json(pdir / "config.json")
        if needs_rebuild_piece(pdir, pcfg):
            return True
        ppdf = pdir / "outputs" / f"{pslug}.pdf"
        times.append(path_mtime(ppdf))
    return out_mtime < max(times) if times else True

def build_programme_pdf(prog_dir: Path, cfg: dict) -> Path:
    outputs = prog_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    out_pdf = outputs / f"{cfg['slug']}.pdf"

    writer = PdfWriter()
    for pslug in cfg.get("pieces", []):
        ppdf = ensure_piece_built(pslug)
        reader = PdfReader(str(ppdf))
        for page in reader.pages:
            writer.add_page(page)

    with out_pdf.open("wb") as f:
        writer.write(f)

    cfg["last_built"] = now_iso()
    save_json(prog_dir / "config.json", cfg)
    return out_pdf

def ensure_programme_built(slug: str) -> Path:
    prog_dir = PROGRAMMES_DIR / slug
    cfg = load_json(prog_dir / "config.json")
    if needs_rebuild_programme(prog_dir, cfg):
        return build_programme_pdf(prog_dir, cfg)
    return prog_dir / "outputs" / f"{cfg['slug']}.pdf"

# ---- Safe file serving and on-demand thumbnails ----

def _is_safe_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False

@app.get("/pieces/<slug>/thumb/<int:page>")
def piece_thumb_page(slug: str, page: int):
    piece_dir = PIECES_DIR / slug
    cfg = load_json(piece_dir / "config.json")
    if cfg.get("type") != "pdf":
        abort(404)
    pdf_path = piece_dir / cfg["source_pdf"]
    if not pdf_path.exists():
        abort(404)
    total = get_pdf_total_pages(pdf_path)
    if page < 1 or page > total:
        abort(404)

    thumbs_dir = piece_dir / "cache" / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    outp = thumbs_dir / f"page-{page}.webp"
    pdf_mtime = path_mtime(pdf_path)

    if (not outp.exists()) or path_mtime(outp) < pdf_mtime:
        images = convert_from_path(str(pdf_path), dpi=100, first_page=page, last_page=page)
        if not images:
            abort(404)
        im = images[0].convert("RGB")
        max_width = 220
        if im.width > max_width:
            ratio = max_width / im.width
            im = im.resize((max_width, max(1, int(im.height * ratio))), Image.LANCZOS)
        im.save(str(outp), format="WEBP", quality=70, method=6)

    if not _is_safe_within(thumbs_dir, outp):
        abort(403)
    resp = send_file(str(outp), mimetype="image/webp", as_attachment=False)
    resp.headers["Cache-Control"] = "public, max-age=604800"
    return resp

@app.get("/pieces/<slug>/source/<path:relpath>")
def piece_source(slug, relpath):
    piece_dir = PIECES_DIR / slug
    fp = (piece_dir / relpath).resolve()
    if not _is_safe_within(piece_dir, fp) or not fp.is_file():
        abort(404)
    return send_file(str(fp), as_attachment=False)

# ---- Routes ----

@app.route("/")
def home():
    ensure_dirs()
    pieces = []
    for d in sorted(PIECES_DIR.glob("*")):
        cfgp = d / "config.json"
        if cfgp.exists():
            pieces.append(load_json(cfgp))
    programmes = []
    for d in sorted(PROGRAMMES_DIR.glob("*")):
        cfgp = d / "config.json"
        if cfgp.exists():
            programmes.append(load_json(cfgp))
    return render_template("home.html", pieces=pieces, programmes=programmes)

# Pieces CRUD + Edit + Preview/Selection

@app.get("/pieces/new")
def new_piece():
    return render_template("new_piece.html")

@app.post("/pieces")
def create_piece():
    ensure_dirs()
    name = request.form["name"].strip()
    ptype = request.form["type"]
    slug = slugify(name)
    piece_dir = PIECES_DIR / slug
    if piece_dir.exists():
        flash("A piece with that name (slug) already exists.", "error")
        return redirect(url_for("new_piece"))
    (piece_dir / "sources").mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": name,
        "slug": slug,
        "type": ptype,
        "metadata": {},
        "updated_at": now_iso(),
        "last_built": None
    }
    if ptype == "pdf":
        f = request.files.get("pdf_file")
        if not f or not f.filename:
            flash("Please upload a PDF.", "error")
            return redirect(url_for("new_piece"))
        pdf_path = piece_dir / "sources" / f.filename
        f.save(str(pdf_path))
        cfg["source_pdf"] = str(pdf_path.relative_to(piece_dir))
        cfg["page_ranges"] = request.form.get("page_ranges", "").strip()
        cfg["selected_pages"] = []
    else:
        files = request.files.getlist("image_files")
        if not files or not files[0].filename:
            flash("Please upload at least one image.", "error")
            return redirect(url_for("new_piece"))
        imgs = []
        for f in files:
            ip = piece_dir / "sources" / f.filename
            f.save(str(ip))
            imgs.append({"path": str(ip.relative_to(piece_dir)), "rotate": 0, "enabled": True})
        cfg["images"] = imgs

    save_json(piece_dir / "config.json", cfg)
    flash("Piece created.", "ok")
    return redirect(url_for("view_piece", slug=slug))

@app.get("/pieces/<slug>")
def view_piece(slug):
    piece_dir = PIECES_DIR / slug
    if not piece_dir.exists():
        abort(404)
    cfg = load_json(piece_dir / "config.json")
    return render_template("piece.html", piece=cfg)

@app.get("/pieces/<slug>/edit")
def edit_piece(slug):
    piece_dir = PIECES_DIR / slug
    if not piece_dir.exists():
        abort(404)
    cfg = load_json(piece_dir / "config.json")
    return render_template("piece_edit.html", piece=cfg)

@app.post("/pieces/<slug>/edit")
def save_piece_edit(slug):
    piece_dir = PIECES_DIR / slug
    if not piece_dir.exists():
        abort(404)
    cfg = load_json(piece_dir / "config.json")
    changed = False

    if cfg["type"] == "pdf":
        pr = request.form.get("page_ranges", "").strip()
        if pr != cfg.get("page_ranges", ""):
            cfg["page_ranges"] = pr
            if pr:
                cfg["selected_pages"] = []
            changed = True

        up = request.files.get("pdf_file")
        new_path_text = request.form.get("source_path", "").strip()
        if up and up.filename:
            dest = piece_dir / "sources" / up.filename
            up.save(str(dest))
            cfg["source_pdf"] = str(dest.relative_to(piece_dir))
            changed = True
        elif new_path_text:
            src = Path(new_path_text).expanduser()
            if src.exists() and src.is_file():
                dest = piece_dir / "sources" / src.name
                shutil.copyfile(src, dest)
                cfg["source_pdf"] = str(dest.relative_to(piece_dir))
                changed = True
            else:
                flash("Source path not found; keeping existing file.", "error")

    else:
        files = request.files.getlist("image_files")
        if files and files[0].filename:
            imgs = cfg.get("images", [])
            for f in files:
                ip = piece_dir / "sources" / f.filename
                f.save(str(ip))
                imgs.append({"path": str(ip.relative_to(piece_dir)), "rotate": 0, "enabled": True})
            cfg["images"] = imgs
            changed = True

    if changed:
        cfg["updated_at"] = now_iso()
        save_json(piece_dir / "config.json", cfg)
        flash("Piece updated.", "ok")
    else:
        flash("No changes to save.", "info")
    return redirect(url_for("view_piece", slug=slug))

@app.post("/pieces/<slug>/generate")
def generate_piece(slug):
    # Kept for compatibility; not exposed in UI
    piece_dir = PIECES_DIR / slug
    if not piece_dir.exists():
        abort(404)
    cfg = load_json(piece_dir / "config.json")
    build_piece_pdf(piece_dir, cfg)
    flash("Piece PDF generated.", "ok")
    return redirect(url_for("view_piece", slug=slug))

@app.get("/pieces/<slug>/pdf")
def piece_pdf(slug):
    pdf_path = ensure_piece_built(slug)
    return send_file(str(pdf_path), mimetype="application/pdf", as_attachment=False, download_name=f"{slug}.pdf")

@app.post("/pieces/<slug>/delete")
def delete_piece(slug):
    piece_dir = PIECES_DIR / slug
    if piece_dir.exists():
        shutil.rmtree(piece_dir)
        flash("Piece deleted.", "ok")
    return redirect(url_for("home"))

# Preview + selection
@app.get("/pieces/<slug>/preview")
def piece_preview(slug):
    piece_dir = PIECES_DIR / slug
    if not piece_dir.exists():
        abort(404)
    cfg = load_json(piece_dir / "config.json")
    if cfg["type"] == "pdf":
        pdf_path = piece_dir / cfg["source_pdf"]
        total = get_pdf_total_pages(pdf_path)
        pages = list(range(1, total + 1))
        selected = set(get_selected_pages_pdf(cfg, total))
        return render_template("piece_preview.html", piece=cfg, pages=pages, selected=selected, mode="pdf")
    else:
        tiles = []
        for i, img in enumerate(cfg.get("images", [])):
            tiles.append({
                "index": i+1,
                "path": img["path"],
                "rotate": int(img.get("rotate", 0)),
                "enabled": bool(img.get("enabled", True))
            })
        return render_template("piece_preview.html", piece=cfg, tiles=tiles, selected=None, mode="images")

@app.post("/pieces/<slug>/pages")
def save_piece_pages(slug):
    piece_dir = PIECES_DIR / slug
    if not piece_dir.exists():
        abort(404)
    cfg = load_json(piece_dir / "config.json")
    if cfg["type"] != "pdf":
        return jsonify({"error": "Not a PDF piece"}), 400
    data = request.get_json(silent=True) or {}
    pages = data.get("pages") or []
    if not isinstance(pages, list):
        return jsonify({"error": "Invalid payload"}), 400
    pdf_path = piece_dir / cfg["source_pdf"]
    total = get_pdf_total_pages(pdf_path)
    pages = sorted({int(p) for p in pages if 1 <= int(p) <= total})
    cfg["selected_pages"] = pages
    cfg["page_ranges"] = pages_to_range_str(pages)
    cfg["updated_at"] = now_iso()
    save_json(piece_dir / "config.json", cfg)
    return jsonify({"ok": True, "selected_pages": pages, "page_ranges": cfg["page_ranges"]})

@app.post("/pieces/<slug>/images/update")
def save_piece_images(slug):
    piece_dir = PIECES_DIR / slug
    if not piece_dir.exists():
        abort(404)
    cfg = load_json(piece_dir / "config.json")
    if cfg["type"] != "images":
        return jsonify({"error": "Not an images piece"}), 400
    data = request.get_json(silent=True) or {}
    items = data.get("items") or []
    new_images = []
    for it in items:
        path = it.get("path")
        if not path:
            continue
        rotate = int(it.get("rotate", 0))
        enabled = bool(it.get("enabled", True))
        new_images.append({"path": path, "rotate": rotate, "enabled": enabled})
    if not new_images:
        return jsonify({"error": "Empty image list"}), 400
    cfg["images"] = new_images
    cfg["updated_at"] = now_iso()
    save_json(piece_dir / "config.json", cfg)
    return jsonify({"ok": True, "count": len(new_images)})

# Programmes

@app.get("/programmes/new")
def new_programme():
    pieces = []
    for d in sorted(PIECES_DIR.glob("*")):
        cfgp = d / "config.json"
        if cfgp.exists():
            pieces.append(load_json(cfgp))
    return render_template("new_programme.html", pieces=pieces)

@app.post("/programmes")
def create_programme():
    name = request.form["name"].strip()
    slug = slugify(name)
    prog_dir = PROGRAMMES_DIR / slug
    if prog_dir.exists():
        flash("A programme with that name (slug) already exists.", "error")
        return redirect(url_for("new_programme"))
    prog_dir.mkdir(parents=True, exist_ok=True)
    pieces = request.form.getlist("pieces")
    cfg = {
        "name": name,
        "slug": slug,
        "pieces": pieces,
        "updated_at": now_iso(),
        "last_built": None
    }
    save_json(prog_dir / "config.json", cfg)
    flash("Programme created.", "ok")
    return redirect(url_for("view_programme", slug=slug))

@app.get("/programmes/<slug>")
def view_programme(slug):
    prog_dir = PROGRAMMES_DIR / slug
    if not prog_dir.exists():
        abort(404)
    cfg = load_json(prog_dir / "config.json")
    resolved = []
    for pslug in cfg.get("pieces", []):
        pcfgp = PIECES_DIR / pslug / "config.json"
        if pcfgp.exists():
            resolved.append(load_json(pcfgp))
    all_pieces = []
    for d in sorted(PIECES_DIR.glob("*")):
        cfgp = d / "config.json"
        if cfgp.exists():
            all_pieces.append(load_json(cfgp))
    return render_template("programme.html", programme=cfg, pieces=resolved, all_pieces=all_pieces)

@app.post("/programmes/<slug>/reorder")
def reorder_programme(slug):
    prog_dir = PROGRAMMES_DIR / slug
    if not prog_dir.exists():
        abort(404)
    cfg = load_json(prog_dir / "config.json")
    data = request.get_json(silent=True) or {}
    new_order = data.get("pieces") or []
    valid = []
    for pslug in new_order:
        if (PIECES_DIR / pslug / "config.json").exists():
            valid.append(pslug)
    if not valid and new_order:
        return jsonify({"error": "No valid pieces"}), 400
    cfg["pieces"] = valid
    cfg["updated_at"] = now_iso()
    save_json(prog_dir / "config.json", cfg)
    return jsonify({"ok": True, "count": len(valid)})

@app.post("/programmes/<slug>/generate")
def generate_programme(slug):
    # Kept for compatibility; not exposed in UI
    prog_dir = PROGRAMMES_DIR / slug
    if not prog_dir.exists():
        abort(404)
    cfg = load_json(prog_dir / "config.json")
    build_programme_pdf(prog_dir, cfg)
    flash("Programme PDF generated.", "ok")
    return redirect(url_for("view_programme", slug=slug))

@app.get("/programmes/<slug>/pdf")
def programme_pdf(slug):
    pdf_path = ensure_programme_built(slug)
    return send_file(str(pdf_path), mimetype="application/pdf", as_attachment=False, download_name=f"{slug}.pdf")

@app.post("/programmes/<slug>/delete")
def delete_programme(slug):
    prog_dir = PROGRAMMES_DIR / slug
    if prog_dir.exists():
        shutil.rmtree(prog_dir)
        flash("Programme deleted.", "ok")
    else:
        flash("Programme not found.", "error")
    return redirect(url_for("home"))

if __name__ == "__main__":
    ensure_dirs()
    app.run(debug=True)
