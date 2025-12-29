#!/usr/bin/env python3

import fitz
import re
import json
import os
import io
import hashlib
from typing import List, Dict, Tuple

PAGE_DROP = "drop"
PAGE_KEEP_REF = "keep_reference"
PAGE_KEEP_EMBED = "keep_embed"

SUBSYSTEM_KEYWORDS = {
    "intake": ["intake", "undertaker", "ground intake"],
    "shooter": ["shooter", "feeder", "flywheel"],
    "climber": ["climb", "climber", "hang"],
    "drivetrain": ["drive", "drivetrain", "swerve", "chassis"],
    "arm": ["arm", "pivot", "linkage"],
    "electrical": ["wiring", "pdh", "pdp", "breaker", "voltage"],
    "software": ["auton", "teleop", "vision", "trajectory", "code"],
}

TOC_REGEX = re.compile(r"\b(table of contents|contents)\b", re.I)
MIN_TOKENS_EMBED = 40


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\-_/]+", text)


def low_information(tokens: List[str]) -> bool:
    if len(tokens) < MIN_TOKENS_EMBED:
        return True
    return len(set(tokens)) / max(len(tokens), 1) < 0.25


def looks_like_divider(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 6:
        return False
    return any(l.isupper() and len(l) > 6 for l in lines)


def detect_subsystem(text: str) -> Tuple[str, float]:
    text_l = text.lower()
    scores = {k: sum(text_l.count(w) for w in v) for k, v in SUBSYSTEM_KEYWORDS.items()}
    best, best_score = max(scores.items(), key=lambda x: x[1])
    total = sum(scores.values())
    if total == 0:
        return ("unknown", 0.0)
    return (best, best_score / total)


def extract_section_headers(page) -> List[str]:
    headers = []
    blocks = page.get_text("dict")["blocks"]

    spans = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                spans.append(span)

    if not spans:
        return headers

    sizes = [s["size"] for s in spans]
    median_size = sorted(sizes)[len(sizes) // 2]

    for span in spans:
        if span["size"] >= median_size * 1.4:
            text = span["text"].strip()
            if 3 <= len(text) <= 80 and (text.isupper() or text.istitle()):
                headers.append(text)

    return list(dict.fromkeys(headers))


class SectionStack:
    def __init__(self):
        self.active_headers: List[str] = []

    def update(self, headers: List[str]):
        if headers:
            self.active_headers = headers

    def current(self) -> List[str]:
        return self.active_headers.copy()


def extract_paragraph_blocks(page) -> List[str]:
    """
    Extract paragraphs from a page using text blocks.

    Returns a list of paragraph dicts:
      {"index": int, "text": str, "tokens": List[str], "bbox": fitz.Rect}
    """
    paragraphs = []
    blocks = page.get_text("dict")["blocks"]

    idx = 0
    for block in blocks:
        if block["type"] != 0:
            continue

        lines = []
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    lines.append(text)

        if not lines:
            continue

        text = normalize_text(" ".join(lines))
        tokens = tokenize(text)
        # discard paragraphs with fewer than 10 tokens
        if len(tokens) < 10:
            idx += 1
            continue

        bbox = None
        # block bbox may be present
        bb = block.get("bbox")
        if bb:
            bbox = fitz.Rect(bb)

        paragraphs.append({
            "index": idx,
            "text": text,
            "tokens": tokens,
            "bbox": bbox,
        })
        idx += 1

    return paragraphs


def chunk_paragraphs(
    paragraphs: List[Dict],
    team: str,
    year: int,
    page_number: int,
    section_headers: List[str],
    subsystem: str,
    subsystem_conf: float,
    max_tokens: int = 250,
) -> List[Dict]:
    """
    Group adjacent mechanism paragraphs into deterministic chunks.

    Paragraphs must be dicts with keys: index, text, tokens, bbox, intent
    """
    chunks = []
    current = []
    current_tokens = 0
    chunk_idx = 0

    def flush_current():
        nonlocal chunk_idx, current, current_tokens
        if not current:
            return
        chunk_text = normalize_text(" ".join(p["text"] for p in current))
        chunk_tokens = sum(len(p["tokens"]) for p in current)
        chunk_id = f"{team}_{year}_p{page_number}_c{chunk_idx}"
        para_indices = [p["index"] for p in current]
        chunks.append({
            "chunk_id": chunk_id,
            "page": page_number,
            "text": chunk_text,
            "subsystem": subsystem,
            "subsystem_confidence": subsystem_conf,
            "section_headers": section_headers,
            "paragraph_indices": para_indices,
            "token_count": chunk_tokens,
        })
        chunk_idx += 1
        current = []
        current_tokens = 0

    for p in paragraphs:
        # only mechanism paragraphs are eligible
        if p.get("intent") != "mechanism":
            # boundary: stop chunk growth at section header boundaries — treat non-mechanism as boundary
            flush_current()
            continue

        p_tokens = len(p["tokens"])
        # if adding would exceed max, flush first
        if current_tokens + p_tokens > max_tokens and current_tokens > 0:
            flush_current()

        current.append(p)
        current_tokens += p_tokens

    flush_current()
    return chunks


def classify_paragraph_intent(text: str) -> str:
    """
    Deterministic heuristic paragraph intent classifier.
    Returns one of: mechanism, requirements, strategy, rules, software, meta
    """
    t = text.lower()
    # mechanism keywords
    mech_kw = ["motor", "gear", "gearbox", "ratio", "mm", "inch", "mount", "mounted", "bolt", "weld", "shaft", "bearing", "plate", "material", "lbs", "kg", "torque"]
    mech_verbs = ["mounted", "powered", "driven", "constructed", "attach", "attach", "drive", "rotate"]
    if any(w in t for w in mech_kw) or any(v in t for v in mech_verbs):
        return "mechanism"

    # software
    sw_kw = ["controller", "state machine", "trajectory", "pid", "vision", "auton", "teleop", "node", "thread", "process"]
    if any(w in t for w in sw_kw):
        return "software"

    # strategy
    strat_kw = ["win", "maximize", "ranking point", "rp", "score", "optimi", "goal", "strategy"]
    if any(w in t for w in strat_kw):
        return "strategy"

    # rules
    rules_kw = ["scoring", "points", "autonomous period", "penalt", "penalty", "foul"]
    if any(w in t for w in rules_kw):
        return "rules"

    # requirements
    req_kw = ["requirement", "shall", "must", "should", "shall not", "shall be"]
    if any(w in t for w in req_kw):
        return "requirements"

    # meta
    meta_kw = ["version", "prepared by", "author", "revision", "table of contents", "contents"]
    if any(w in t for w in meta_kw):
        return "meta"

    # default conservative
    return "meta"


def extract_images_and_anchor(doc, page, page_number: int, mechanism_paragraphs: List[Dict], images_out_dir: str) -> List[Dict]:
    """
    Extract images from a page and deterministically anchor them to nearest mechanism paragraph.

    Returns list of image metadata dicts.
    """
    images_meta = []
    blocks = page.get_text("dict").get("blocks", [])

    # build paragraph bbox list for anchoring
    paras = [p for p in mechanism_paragraphs if p.get("bbox") is not None]

    for block in blocks:
        if block.get("type") != 1:
            continue
        bbox = fitz.Rect(block.get("bbox"))
        img_info = block.get("image")
        # xref may be available in different shapes depending on fitz version
        xref = None
        if isinstance(img_info, dict):
            xref = img_info.get("xref") or block.get("xref")
        elif isinstance(img_info, (bytes, bytearray)):
            xref = block.get("xref")
        else:
            xref = block.get("xref")

        img_bytes = None
        # try to extract via xref
        try:
            if xref:
                img_dict = doc.extract_image(xref)
                img_bytes = img_dict.get("image")
        except Exception:
            img_bytes = None

        # fallback: rasterize clipped area to get image bytes
        if not img_bytes:
            try:
                pix = page.get_pixmap(clip=bbox)
                img_bytes = pix.tobytes("png")
            except Exception:
                img_bytes = None

        if not img_bytes:
            continue

        # deterministic image id from sha1 of bytes
        h = hashlib.sha1(img_bytes).hexdigest()[:16]
        image_id = f"img_{h}_p{page_number}_x{int(bbox.x0)}_y{int(bbox.y0)}"

        os.makedirs(images_out_dir, exist_ok=True)
        out_path = os.path.join(images_out_dir, f"{image_id}.png")
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        # anchor to nearest mechanism paragraph (prefer above)
        anchor_para = None
        best_dist = None
        page_h = page.rect.height
        max_dist = max(page_h * 0.25, 100)

        for p in paras:
            pb = p["bbox"]
            # compute vertical distance where positive means paragraph below image
            if pb.y1 <= bbox.y0:
                dist = bbox.y0 - pb.y1
            elif pb.y0 >= bbox.y1:
                dist = pb.y0 - bbox.y1
            else:
                dist = 0

            if best_dist is None or dist < best_dist:
                best_dist = dist
                anchor_para = p

        anchored = False
        anchor_par_index = None
        if anchor_para and best_dist is not None and best_dist <= max_dist:
            anchored = True
            anchor_par_index = anchor_para["index"]

        images_meta.append({
            "image_id": image_id,
            "source_page": page_number,
            "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
            "file": out_path,
            "anchored": anchored,
            "anchor_paragraph_index": anchor_par_index,
            "caption": None,
            "visual_facts": [],
        })

    return images_meta


def preprocess_pdf(pdf_path: str, team: str, year: int) -> Dict:
    doc = fitz.open(pdf_path)
    section_stack = SectionStack()

    pages_out = []
    pages_to_keep = []

    for i, page in enumerate(doc):
        raw_text = page.get_text()
        text = normalize_text(raw_text)
        tokens = tokenize(text)

        page_meta = {
            "page_number": i + 1,
            "state": PAGE_KEEP_EMBED,
            "embed": True,
            "reasons": [],
            "team": team,
            "year": year,
        }

        if i == 0 or TOC_REGEX.search(text):
            page_meta["state"] = PAGE_DROP
            page_meta["embed"] = False
            page_meta["reasons"].append("cover_or_toc")
            pages_out.append(page_meta)
            continue

        if looks_like_divider(text):
            page_meta["state"] = PAGE_KEEP_REF
            page_meta["embed"] = False
            page_meta["reasons"].append("divider_page")

        elif low_information(tokens):
            page_meta["state"] = PAGE_KEEP_REF
            page_meta["embed"] = False
            page_meta["reasons"].append("low_information")

        headers = extract_section_headers(page)
        section_stack.update(headers)

        page_meta["section_headers"] = headers
        page_meta["active_section_path"] = section_stack.current()

        subsystem, conf = detect_subsystem(text)
        page_meta["subsystem"] = subsystem
        page_meta["subsystem_confidence"] = conf
        page_meta["token_count"] = len(tokens)

        paragraphs = extract_paragraph_blocks(page)

        # classify paragraph intents deterministically
        for p in paragraphs:
            p["intent"] = classify_paragraph_intent(p["text"])

        # attach paragraph metadata to page
        page_meta["paragraphs"] = paragraphs

        # chunk mechanism paragraphs into deterministic chunks
        chunks = chunk_paragraphs(
            paragraphs,
            team=team,
            year=year,
            page_number=i + 1,
            section_headers=section_stack.current(),
            subsystem=subsystem,
            subsystem_conf=conf,
            max_tokens=250,
        )

        page_meta["chunks"] = chunks

        pages_out.append(page_meta)

        if page_meta["state"] != PAGE_DROP and len(tokens) >= 5:
            pages_to_keep.append(i)

    return {
        "team": team,
        "year": year,
        "pages": pages_out,
        "pages_to_keep": pages_to_keep,
        "doc": doc,
    }


if __name__ == "__main__":
    input_pdf = "/home/aadi/L-Projects/frc-rag-again/backend/data/254-2025.pdf"
    output_dir = "processed"

    result = preprocess_pdf(input_pdf, team="254", year=2025)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare metadata structures and extract images/associate
    images_out_dir = os.path.join(output_dir, "images")
    all_images = []

    # For each page, extract images and anchor to mechanism paragraphs
    for page_meta in result["pages"]:
        page_num = page_meta["page_number"]
        src_page = result["doc"][page_num - 1]
        mech_pars = [p for p in page_meta.get("paragraphs", []) if p.get("intent") == "mechanism"]
        imgs = extract_images_and_anchor(result["doc"], src_page, page_num, mech_pars, images_out_dir)
        page_meta["images"] = imgs
        all_images.extend(imgs)

        # map paragraph index -> chunk id for this page
        par_to_chunk = {}
        for c in page_meta.get("chunks", []):
            # ensure images list exists on chunk
            c.setdefault("images", [])
            for pidx in c.get("paragraph_indices", []):
                par_to_chunk[pidx] = c["chunk_id"]

        # propagate anchored images to chunks
        for im in imgs:
            if im.get("anchored") and im.get("anchor_paragraph_index") is not None:
                cid = par_to_chunk.get(im["anchor_paragraph_index"])
                if cid:
                    im["chunk_id"] = cid
                    # find chunk and append image id
                    for c in page_meta.get("chunks", []):
                        if c["chunk_id"] == cid:
                            c.setdefault("images", []).append(im["image_id"])
                            break
                else:
                    im["chunk_id"] = None

    # Write final metadata combining pages, chunks, images
    metadata = {
        "team": result["team"],
        "year": result["year"],
        "pages": result["pages"],
        "images": all_images,
        "total_pages_original": len(result["pages"]),
        "total_pages_kept": len(result["pages_to_keep"]),
    }

    # convert any non-serializable objects (fitz.Rect) to lists on a copy
    serializable = {
        "team": metadata["team"],
        "year": metadata["year"],
        "pages": [],
        "images": metadata.get("images", []),
        "total_pages_original": metadata.get("total_pages_original"),
        "total_pages_kept": metadata.get("total_pages_kept"),
    }

    for pg in metadata.get("pages", []):
        pg_copy = dict(pg)
        paras = []
        for p in pg.get("paragraphs", []) or []:
            pcopy = dict(p)
            bb = pcopy.get("bbox")
            if bb is None:
                pcopy["bbox"] = None
            else:
                pcopy["bbox"] = [bb.x0, bb.y0, bb.x1, bb.y1]
            paras.append(pcopy)
        pg_copy["paragraphs"] = paras
        # images are already serializable
        serializable["pages"].append(pg_copy)

    json_path = os.path.join(output_dir, "processed_binder.json")
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Metadata saved to: {json_path}")

    # create filtered PDF (kept pages)
    filtered_pdf = fitz.open()
    for page_idx in result["pages_to_keep"]:
        filtered_pdf.insert_pdf(result["doc"], from_page=page_idx, to_page=page_idx)

    filtered_pdf_path = os.path.join(output_dir, os.path.basename(input_pdf))
    filtered_pdf.save(filtered_pdf_path)
    filtered_pdf.close()

    # create chunks PDF: one combined page per original source page.
    # For each source page, union all chunk paragraph bboxes and associated image bboxes,
    # render that region once, and list all chunk IDs belonging to the page in the footer.
    chunks_pdf = fitz.open()
    for page_meta in result["pages"]:
        page_num = page_meta["page_number"]
        src_page = result["doc"][page_num - 1]

        # collect all bboxes for paragraphs included in any chunk on this page
        clip_bboxes = []
        chunk_ids = []
        for chunk in page_meta.get("chunks", []):
            chunk_ids.append(chunk.get("chunk_id"))
            for pidx in chunk.get("paragraph_indices", []):
                for p in page_meta.get("paragraphs", []):
                    if p["index"] == pidx and p.get("bbox") is not None:
                        clip_bboxes.append(p["bbox"])
                        break

        # include any images anchored to any chunk on this page
        for im in page_meta.get("images", []) or []:
            ib = im.get("bbox")
            if ib:
                try:
                    rect = fitz.Rect(ib)
                except Exception:
                    rect = None
                if rect is not None:
                    clip_bboxes.append(rect)

        if clip_bboxes:
            clip = clip_bboxes[0]
            for r in clip_bboxes[1:]:
                clip |= r
            pad = 8
            clip_rect = fitz.Rect(
                max(src_page.rect.x0, clip.x0 - pad),
                max(src_page.rect.y0, clip.y0 - pad),
                min(src_page.rect.x1, clip.x1 + pad),
                min(src_page.rect.y1, clip.y1 + pad),
            )
        else:
            clip_rect = src_page.rect

        scale = 2.0
        matrix = fitz.Matrix(scale, scale)

        try:
            pix = src_page.get_pixmap(clip=clip_rect, matrix=matrix)
        except Exception:
            pix = src_page.get_pixmap(matrix=matrix)
            clip_rect = src_page.rect

        page_w = pix.width
        page_h = pix.height

        p = chunks_pdf.new_page(width=page_w, height=page_h)
        img_stream = pix.tobytes("png")
        p.insert_image(fitz.Rect(0, 0, page_w, page_h), stream=img_stream)

        # footer: list chunk ids (comma separated) or single id
        footer_text = ", ".join(chunk_ids) if chunk_ids else ""
        footer_y = max(page_h - 18, page_h - 40)
        p.insert_text((10, footer_y), footer_text, fontsize=10)

    chunks_pdf_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(input_pdf))[0] + "_chunks.pdf"
    )
    chunks_pdf.save(chunks_pdf_path)
    chunks_pdf.close()
    result["doc"].close()

    print(f"Filtered PDF saved to: {filtered_pdf_path}")
    print(f"Chunks PDF saved to: {chunks_pdf_path}")
