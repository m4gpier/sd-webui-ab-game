"""
Prompt Evolution — A1111 Extension
====================================
A/B…N image game driven by genetic-style prompt mutation.
2–9 choices per round. Winner mutates into N-1 children next round.
"""

import os
import random
import datetime
import base64
import io
import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks, shared

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tag_scores


# ---------------------------------------------------------------------------
# Module-level stop flag — survives across Gradio generator yields
# ---------------------------------------------------------------------------

_stop_requested = False


def request_stop():
    global _stop_requested
    _stop_requested = True
    try:
        shared.state.interrupt()
    except Exception:
        pass


def clear_stop():
    global _stop_requested
    _stop_requested = False
    try:
        shared.state.interrupted = False
        shared.state.skipped = False
    except Exception:
        pass


def is_stop_requested() -> bool:
    return _stop_requested


# ---------------------------------------------------------------------------
# A1111 Script stub
# ---------------------------------------------------------------------------

class Script(scripts.Script):
    def title(self):
        return "The A/B Game"

    def show(self, is_img2img):
        return False


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def get_available_loras() -> list[str]:
    import pathlib
    base = pathlib.Path(shared.data_path)
    candidates = [
        base / "models" / "Lora",
        base / "models" / "LyCORIS",
    ]
    extra = getattr(shared.opts, "lora_dir", None)
    if extra:
        candidates.append(pathlib.Path(extra))
    names = set()
    for d in candidates:
        if d.exists():
            for ext in ("*.safetensors", "*.pt", "*.ckpt"):
                for f in d.rglob(ext):
                    names.add(f.stem)
    return sorted(names, key=str.lower)


def apply_loras_to_prompt(prompt: str, lora_selections: list[str], lora_weight: float) -> str:
    if not lora_selections:
        return prompt
    tags = " ".join(f"<lora:{n}:{lora_weight:.2f}>" for n in lora_selections)
    return f"{prompt}, {tags}" if prompt.strip() else tags


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

_vocab_cache: dict[str, list[str]] = {}


def find_danbooru_csv() -> str | None:
    import pathlib
    our_dir = pathlib.Path(__file__).parent.parent
    extensions_root = our_dir.parent
    for name in ("a1111-sd-webui-tagcomplete", "sd-webui-tagcomplete", "tagcomplete"):
        p = extensions_root / name / "tags" / "danbooru.csv"
        if p.exists():
            return str(p)
    return None


def build_clip_vocab_with_progress(freq_min: float, freq_max: float):
    checkpoint_info = getattr(shared.sd_model, "sd_checkpoint_info", None)
    model_title = (getattr(checkpoint_info, "title", None)
                   or getattr(checkpoint_info, "name", None)
                   or str(checkpoint_info))
    model_key = f"{model_title}_{freq_min}_{freq_max}"

    if model_key in _vocab_cache:
        yield "✅ Vocab ready (cached)", 1.0, _vocab_cache[model_key]
        return

    yield "🔍 Locating CLIP tokenizer…", 0.0, None
    try:
        model = shared.sd_model
        if hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'tokenizer'):
            tokenizer = model.cond_stage_model.tokenizer
        elif hasattr(model, 'conditioner'):
            tokenizer = None
            for embedder in model.conditioner.embedders:
                if hasattr(embedder, 'tokenizer'):
                    tokenizer = embedder.tokenizer
                    break
        else:
            raise Exception("Could not locate tokenizer in this model architecture")
        if tokenizer is None:
            raise Exception("No tokenizer found in conditioner embedders")
        clip_vocab = set(
            t.replace("</w>", "").strip().lower()
            for t in tokenizer.get_vocab().keys()
        )
    except Exception as e:
        yield f"⚠️ {e}", 1.0, []
        return

    import math
    lo = 10 ** min(freq_min, freq_max)
    hi = 10 ** max(freq_min, freq_max)

    danbooru_path = find_danbooru_csv()
    if danbooru_path:
        yield "📖 Loading Danbooru tags…", 0.1, None
        try:
            import csv
            tags = []
            with open(danbooru_path, "r", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) < 3:
                        continue
                    try:
                        tag_type = int(row[1])
                        post_count = int(row[2])
                    except ValueError:
                        continue
                    if tag_type != 0 or not (lo <= post_count <= hi):
                        continue
                    name = row[0].strip()
                    # Keep multi-word tags as-is ("long_hair" -> "long hair").
                    # No CLIP-vocab filter: the model knows danbooru tags as
                    # phrases; requiring single-token matches wrongly dropped
                    # nearly all 2+ word tags.
                    tags.append(name.replace("_", " "))
            if tags:
                _vocab_cache[model_key] = tags
                yield f"✅ Vocab ready — {len(tags):,} danbooru tags", 1.0, tags
                return
        except Exception as e:
            print(f"[prompt-evolution] Danbooru CSV failed: {e}, falling back to wordfreq")

    yield "📖 Loading wordfreq table…", 0.1, None
    try:
        from wordfreq import get_frequency_dict
    except ImportError:
        yield "⚠️ wordfreq not installed — using fallback vocab", 1.0, ["art", "portrait", "landscape", "fantasy", "vivid"]
        return

    freq_dict = get_frequency_dict("en", wordlist="best")
    freq_lo, freq_hi = min(freq_min, freq_max), max(freq_min, freq_max)
    def to_zipf(v):
        return math.log10(v) + 9 if v > 0 else 0

    yield "🧬 Filtering vocab…", 0.2, None
    clean = []
    for token in tokenizer.get_vocab().keys():
        t = token.replace("</w>", "").strip()
        if len(t) >= 3 and t.isalpha():
            zipf = to_zipf(freq_dict.get(t, 0))
            if freq_lo <= zipf <= freq_hi:
                clean.append(t)

    _vocab_cache[model_key] = clean
    yield f"✅ Vocab ready — {len(clean):,} tokens (wordfreq)", 1.0, clean


def get_clip_vocab(freq_min: float, freq_max: float) -> list[str]:
    checkpoint_info = getattr(shared.sd_model, "sd_checkpoint_info", None)
    model_title = (getattr(checkpoint_info, "title", None)
                   or getattr(checkpoint_info, "name", None)
                   or str(checkpoint_info))
    model_key = f"{model_title}_{freq_min}_{freq_max}"
    if model_key in _vocab_cache:
        return _vocab_cache[model_key]
    for _, _, vocab in build_clip_vocab_with_progress(freq_min, freq_max):
        if vocab is not None:
            return vocab
    return []


# ---------------------------------------------------------------------------
# Prompt mutation
# ---------------------------------------------------------------------------

_COLOR_PART_CHANCE = 0.25  # probability of picking a colour+part compound vs danbooru tag


def _pick_token(vocab: list[str], color_part_vocab: list[str], bias: float,
                pool_tags: list[str], pool_weights: list[float],
                scores: dict[str, float] | None = None,
                explore: float = 0.0) -> str:
    """
    Choose a new token.
    - With probability `bias`, draw from the score-weighted pool (positive tags only).
    - Otherwise: 75% danbooru, 25% colour+part compound, with rejection sampling
      that suppresses negatively-scored tags. A tag with score s has pass probability
      1/(1+|s|), so score −1 → 50%, −3 → 25%, −10 → ~9%, −20 → ~5%.
      Unscored tags (score 0) always pass.
    Falls back gracefully if either vocab is empty.
    """
    def _passes(tok: str) -> bool:
        if not scores:
            return True
        # Check the token itself and its expanded components (colour/part)
        parts = tok.strip().lower().split()
        worst = 0.0
        for p in ([tok.strip().lower()] + (parts if len(parts) == 2 else [])):
            s = scores.get(p, 0)
            if s < worst:
                worst = s
        if worst >= 0:
            return True
        return random.random() < 1.0 / (1.0 + abs(worst))

    if bias > 0 and pool_tags and random.random() < bias:
        return random.choices(pool_tags, weights=pool_weights, k=1)[0]

    # Exploration bonus: with probability `explore`, prefer a never-scored token
    if explore > 0 and scores and random.random() < explore:
        for _ in range(10):
            if color_part_vocab and (not vocab or random.random() < _COLOR_PART_CHANCE):
                tok = random.choice(color_part_vocab)
            else:
                tok = random.choice(vocab) if vocab else random.choice(color_part_vocab)
            if tok.strip().lower() not in scores:
                return tok

    # Random draw with suppression — try up to 10 times, fall back if nothing passes
    for _ in range(10):
        if color_part_vocab and (not vocab or random.random() < _COLOR_PART_CHANCE):
            tok = random.choice(color_part_vocab)
        else:
            tok = random.choice(vocab) if vocab else random.choice(color_part_vocab)
        if _passes(tok):
            return tok

    # Fallback: return whatever we got last (extremely rare, avoids infinite loop)
    return tok


# ---------------------------------------------------------------------------
# Color + body-part compound tags
# ---------------------------------------------------------------------------

_COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "white",
    "black", "grey", "gray", "brown", "blonde", "silver", "gold", "cyan",
    "magenta", "violet", "teal", "turquoise", "dark", "light", "bright",
}

_PARTS = {
    "eyes", "hair", "lips", "skin", "shirt", "dress", "jacket", "coat",
    "pants", "skirt", "shoes", "boots", "gloves", "hat", "scarf", "tie",
    "wings", "tail", "ears", "horns", "nails", "eyebrows", "eyelashes",
    "hoodie", "cape", "ribbon", "bow", "socks", "stockings", "leggings",
}


def build_color_part_vocab() -> list[str]:
    """Generate all color+part combinations (e.g. 'red eyes', 'blue hair')."""
    return [f"{c} {p}" for c in _COLORS for p in _PARTS]


_color_part_vocab: list[str] | None = None


def get_color_part_vocab() -> list[str]:
    global _color_part_vocab
    if _color_part_vocab is None:
        _color_part_vocab = build_color_part_vocab()
    return _color_part_vocab


def _extract_element(tag: str) -> str | None:
    """Return the body-part element from a color+part tag, or None."""
    tag_l = tag.strip().lower()
    for part in _PARTS:
        if tag_l.endswith(" " + part) or tag_l == part:
            return part
    return None


def _extract_color(tag: str) -> str | None:
    """Return the colour from a color+part tag, or None."""
    tag_l = tag.strip().lower()
    parts = tag_l.split()
    if len(parts) == 2 and parts[0] in _COLORS and parts[1] in _PARTS:
        return parts[0]
    return None


MAX_COLORS = 3
MAX_PARTS  = 3


def _deduplicate_tokens(tokens: list[str]) -> list[str]:
    """
    Remove duplicate element conflicts: if 'red eyes' and 'blue eyes' both
    exist, keep only the last one encountered (the newer mutation wins).
    Also removes exact duplicates.
    """
    seen_elements: dict[str, int] = {}  # element -> last index
    seen_exact: set[str] = set()

    remove = set()
    for i, tok in enumerate(tokens):
        tok_l = tok.strip().lower()
        if tok_l in seen_exact:
            remove.add(i)
            continue
        seen_exact.add(tok_l)
        elem = _extract_element(tok_l)
        if elem is not None:
            if elem in seen_elements:
                remove.add(seen_elements[elem])
            seen_elements[elem] = i

    return [t for i, t in enumerate(tokens) if i not in remove]


def _enforce_caps(tokens: list[str]) -> list[str]:
    """
    Enforce max 3 distinct colours and 3 distinct parts across all
    colour+part compounds. When over the cap, remove the earliest
    offending compound (newest mutations are appended last, so we
    preserve the most recent choices).
    """
    for cap, extractor in [(MAX_COLORS, _extract_color), (MAX_PARTS, _extract_element)]:
        while True:
            seen: dict[str, int] = {}   # value -> first index among compounds
            for i, tok in enumerate(tokens):
                tok_l = tok.strip().lower()
                # Only consider colour+part compounds for cap tracking
                if _extract_color(tok_l) is None:
                    continue
                val = extractor(tok_l)
                if val and val not in seen:
                    seen[val] = i
            if len(seen) <= cap:
                break
            # Remove the compound at the earliest index (oldest, least-recently mutated)
            oldest_idx = min(seen.values())
            tokens = [t for i, t in enumerate(tokens) if i != oldest_idx]

    return tokens


def _would_exceed_cap(new_token: str, tokens: list[str]) -> bool:
    """
    Return True if adding new_token would push colours or parts over their cap,
    and there's no existing compound with the same colour/part to be displaced
    by deduplication (meaning the cap hit is genuinely new).
    """
    tok_l = new_token.strip().lower()
    colour = _extract_color(tok_l)
    part   = _extract_element(tok_l)
    if colour is None:
        return False   # not a compound, never capped

    existing_colours = {_extract_color(t.strip().lower()) for t in tokens if _extract_color(t.strip().lower())}
    existing_parts   = {_extract_element(t.strip().lower()) for t in tokens if _extract_color(t.strip().lower())}

    # If same part already exists it'll be deduped (swapped), not added — no cap hit
    if part in existing_parts:
        return False

    colour_hit = colour not in existing_colours and len(existing_colours) >= MAX_COLORS
    part_hit   = len(existing_parts) >= MAX_PARTS
    return colour_hit or part_hit


def _pick_swap_idx(mutable: list[int], tokens: list[str], bias: float,
                   bad_tags: set[str]) -> int:
    """
    Choose which mutable slot to swap. With probability `bias`, prefer slots
    whose token is in bad_tags; fall back to uniform if none qualify.
    """
    if bias > 0 and bad_tags:
        bad_idxs = [i for i in mutable if tokens[i].strip().lower() in bad_tags]
        if bad_idxs and random.random() < bias:
            return random.choice(bad_idxs)
    return random.choice(mutable)


def mutate_prompt(prompt: str, vocab: list[str], mutation_rate: int,
                  locked: list[str] = None, max_tags: int = 75,
                  bias: float = 0.0,
                  pool_tags: list[str] = None, pool_weights: list[float] = None,
                  bad_tags: set[str] = None, scores: dict = None,
                  explore: float = 0.0) -> str:
    tokens = [t.strip() for t in prompt.split(",") if t.strip()]
    locked_set = set(t.strip().lower() for t in locked) if locked else set()
    pool_tags = pool_tags or []
    pool_weights = pool_weights or []
    bad_tags = bad_tags or set()

    color_part_vocab = get_color_part_vocab()

    for _ in range(mutation_rate):
        if not vocab and not color_part_vocab:
            break
        new_token = _pick_token(vocab, color_part_vocab, bias, pool_tags, pool_weights,
                                scores=scores, explore=explore)

        # If new_token is a compound that would genuinely exceed colour/part caps
        # (and won't just displace an existing same-part token), skip it
        if _would_exceed_cap(new_token, tokens):
            continue

        mutable = [i for i, t in enumerate(tokens) if t.lower() not in locked_set]

        token_count = len(tokens)
        if token_count >= max_tags:
            add_chance = 0.0
        else:
            add_chance = 1.0 - (token_count / max_tags)

        # If any mutable token is bad-scored, bias forces a swap regardless of add_chance
        has_bad = any(tokens[i].strip().lower() in bad_tags for i in mutable)
        forced_swap = has_bad and bias > 0 and random.random() < bias

        if mutable and (forced_swap or (not (random.random() < add_chance))):
            idx = _pick_swap_idx(mutable, tokens, bias, bad_tags)
            tokens[idx] = new_token
        else:
            tokens.append(new_token)

    # Deduplicate then enforce colour/part caps
    tokens = _deduplicate_tokens(tokens)
    tokens = _enforce_caps(tokens)

    return ", ".join(tokens)


def seed_prompt(player_prompt: str, vocab: list[str]) -> str:
    if player_prompt.strip():
        return player_prompt.strip()
    if vocab:
        return random.choice(vocab)
    return "a beautiful painting"


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def get_save_dir() -> str:
    base = shared.opts.outdir_txt2img_samples
    save_dir = os.path.join(base, "prompt-evolution")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def get_game_dir(game_num: int) -> str:
    """Folder structure: prompt-evolution/YYYYMMDD/G{num:03d}/"""
    today = datetime.datetime.now().strftime("%Y%m%d")
    d = os.path.join(get_save_dir(), today, f"G{game_num:03d}")
    os.makedirs(d, exist_ok=True)
    return d


def resolve_game_num() -> int:
    """
    Scan the save directory for files matching today's date prefix and return
    the next game number. Looks at all files named YYYYMMDD_G{N}_* and returns
    max(N) + 1, or 1 if none exist today.
    Persists across sessions since it reads from disk each time.
    """
    import re
    today = datetime.datetime.now().strftime("%Y%m%d")
    day_dir = os.path.join(get_save_dir(), today)
    pattern = re.compile(r"^G(\d+)$")
    max_game = 0
    try:
        for fname in os.listdir(day_dir):
            m = pattern.match(fname)
            if m and os.path.isdir(os.path.join(day_dir, fname)):
                max_game = max(max_game, int(m.group(1)))
    except Exception:
        pass
    return max_game + 1


def save_images(imgs: list, labels: list[str], game_num: int, round_num: int) -> list[str]:
    """
    Save into prompt-evolution/YYYYMMDD/G{game:03d}/ with names
    HHMMSS_R{round:03d}_{label}.png (time-marker titles inside the game folder).
    """
    paths = []
    try:
        game_dir = get_game_dir(game_num)
        stamp = datetime.datetime.now().strftime("%H%M%S")
        for img, label in zip(imgs, labels):
            fname = f"{stamp}_R{round_num:03d}_{label}.png"
            path = os.path.join(game_dir, fname)
            if img is not None:
                img.save(path)
                paths.append(path)
    except Exception as e:
        print(f"[prompt-evolution] Save failed: {e}")
    return paths


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def pil_to_data_uri(img) -> str:
    if img is None:
        return ""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def pil_to_thumbnail_uri(img, size: int = 120) -> str:
    if img is None:
        return ""
    thumb = img.copy()
    thumb.thumbnail((size, size))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=75, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Image panel HTML — N images in a responsive grid
# ---------------------------------------------------------------------------

LABELS = list("ABCDEFGHI")  # up to 9


# Image height cap per card (px)
IMG_MAX_H = 320


def build_image_card_html(img, prompt: str, width: int, height: int,
                           cfg: float, seed: int, generating: bool = False,
                           test_type: str | None = None) -> str:
    """Single image card HTML — no buttons (those are Gradio components beneath).
    test_type: 'neg', 'pos', or None — shows a small sticker corner badge."""
    seed_str = str(seed) if seed != -1 else "random"
    if img is not None:
        uri = pil_to_data_uri(img)
        img_tag = (
            f"<img src='{uri}' style='width:100%;max-height:{IMG_MAX_H}px;"
            f"object-fit:contain;border-radius:6px;display:block;' />"
        )
    elif generating:
        img_tag = (
            f"<div style='width:100%;height:{IMG_MAX_H}px;background:#1a1a2e;"
            f"border-radius:6px;display:flex;align-items:center;justify-content:center;"
            f"font-size:12px;color:#aaa;'>generating…</div>"
        )
    else:
        img_tag = (
            f"<div style='width:100%;height:{IMG_MAX_H}px;background:#111;"
            f"border-radius:6px;'></div>"
        )

    if test_type == "neg":
        sticker = ("<div style='position:absolute;top:4px;right:4px;background:#c0392b;"
                   "color:#fff;font-size:9px;font-weight:700;padding:2px 5px;"
                   "border-radius:4px;pointer-events:none;'>🧪 NEG</div>")
    elif test_type == "pos":
        sticker = ("<div style='position:absolute;top:4px;right:4px;background:#27ae60;"
                   "color:#fff;font-size:9px;font-weight:700;padding:2px 5px;"
                   "border-radius:4px;pointer-events:none;'>🧪 POS</div>")
    else:
        sticker = ""

    img_wrap = (f"<div style='position:relative;width:100%;'>{img_tag}{sticker}</div>"
                if sticker else img_tag)

    meta = (
        f"<div style='font-size:9px;color:#666;margin-top:3px;text-align:center;'>"
        f"{width}×{height} · CFG {cfg} · {seed_str}</div>"
    )
    return img_wrap + meta


# ---------------------------------------------------------------------------
# Lineage HTML — one row per round, N thumbnails, winner blue-highlighted
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Timeline image (family tree of the whole game)
# ---------------------------------------------------------------------------

def _decode_thumb(uri: str):
    """Decode a data-URI thumbnail back to a PIL image (or None)."""
    import base64, io
    from PIL import Image
    try:
        b64 = uri.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        return None


def generate_timeline_image(lineage: list[dict], starting_tokens: list[str],
                            game_num: int = 1) -> str | None:
    """
    Render the whole game as one PNG family tree: full-size images with seed,
    full prompt (new-vs-parent tags highlighted, never truncated), reactions,
    test badges, winner highlight, and lines from each winner to the next round.
    Returns the saved file path, or None if there's nothing to draw.
    """
    from PIL import Image, ImageDraw, ImageFont
    if not lineage:
        return None

    try:
        font  = ImageFont.truetype("DejaVuSans.ttf", 16)
        fontb = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = fontb = ImageFont.load_default()

    PAD, CPAD, ROW_GAP, LINE_H = 32, 14, 56, 22

    BG, FG, DIM = (24, 24, 26), (220, 220, 220), (140, 140, 145)
    GREEN, RED, GOLD = (110, 200, 120), (235, 110, 100), (250, 200, 80)
    BLUE, PURPLE = (110, 170, 240), (190, 130, 230)
    start_set = {t.strip().lower() for t in starting_tokens}

    def get_img(entry, col):
        paths = entry.get("img_paths") or []
        if col < len(paths) and paths[col] and os.path.exists(paths[col]):
            try:
                return Image.open(paths[col]).convert("RGB")
            except Exception:
                pass
        return None  # -> "Image Not Available" placeholder

    # Cell width = widest image in the game + padding
    img_w = img_h_default = 150
    for e in lineage:
        for c in range(len(e["prompts"])):
            im = get_img(e, c)
            if im:
                img_w = max(img_w, im.width)
    CELL_W = img_w + CPAD * 2

    # measure = ImageDraw on a scratch image
    scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def wrap_prompt(prompt, parent):
        """Return list of lines, each a list of (text, color)."""
        parent_tags = {t.strip().lower() for t in parent.split(",") if t.strip()}
        lines, cur, cur_w = [], [], 0
        max_w = CELL_W - CPAD * 2
        for tag in [t.strip() for t in prompt.split(",") if t.strip()]:
            tl = tag.lower()
            col = DIM if tl in start_set else (GREEN if tl not in parent_tags else FG)
            piece = tag + ", "
            wpx = scratch.textlength(piece, font=font)
            if cur and cur_w + wpx > max_w:
                lines.append(cur)
                cur, cur_w = [], 0
            cur.append((piece, col))
            cur_w += wpx
        if cur:
            lines.append(cur)
        return lines

    # ── Pass 1: measure every cell, derive per-row heights ────────────────
    rows = []
    prev_parent = ""
    for entry in lineage:
        n = len(entry["prompts"])
        cells = []
        max_h = 0
        for c in range(n):
            im = get_img(entry, c)
            ih = im.height if im else img_h_default
            plines = wrap_prompt(entry["prompts"][c], prev_parent)
            text_h = LINE_H + len(plines) * LINE_H + 10   # seed line + prompt lines
            cell_h = 10 + ih + 8 + text_h + 10
            cells.append({"img": im, "img_h": ih, "plines": plines, "cell_h": cell_h})
        max_h = max(c["cell_h"] for c in cells)
        rows.append({"entry": entry, "cells": cells, "row_h": max_h})
        prev_parent = entry["prompts"][entry.get("winner_idx", 0)]

    max_n = max(len(r["cells"]) for r in rows)
    W = PAD * 2 + max_n * CELL_W
    H = PAD * 2 + sum(r["row_h"] + ROW_GAP for r in rows)

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ── Pass 2: draw ──────────────────────────────────────────────────────
    prev_winner_center = None
    y = PAD
    for row in rows:
        entry, cells = row["entry"], row["cells"]
        n        = len(cells)
        widx     = entry.get("winner_idx", 0)
        seed     = entry.get("seed", -1)
        reacts   = entry.get("reactions", [None] * n)
        neg_slot = entry.get("neg_test_slot")
        pos_slot = entry.get("pos_test_slot")

        draw.text((PAD, y - 22), f"Round {entry.get('round', '?')}", font=fontb, fill=DIM)
        x_base = PAD + (W - 2 * PAD - n * CELL_W) // 2

        centers = []
        for c, cell in enumerate(cells):
            x0 = x_base + c * CELL_W
            cx_mid = x0 + CELL_W // 2
            centers.append((cx_mid, y, y + row["row_h"]))

            border = GOLD if c == widx else (45, 45, 50)
            draw.rounded_rectangle([x0 + 4, y, x0 + CELL_W - 4, y + row["row_h"]],
                                   radius=8, outline=border, width=3 if c == widx else 1)

            im = cell["img"]
            tx = x0 + (CELL_W - (im.width if im else img_w)) // 2
            ty = y + 10
            if im:
                img.paste(im, (tx, ty))
            else:
                draw.rectangle([tx, ty, tx + img_w, ty + cell["img_h"]], fill=(40, 40, 44))
                msg = "Image Not Available"
                mw = draw.textlength(msg, font=font)
                draw.text((tx + (img_w - mw) // 2, ty + cell["img_h"] // 2 - 8),
                          msg, font=font, fill=DIM)

            badge_y = ty + 6
            if c < len(reacts) and reacts[c] == 1:
                draw.text((tx + 6, badge_y), "LIKE", font=fontb, fill=GREEN)
            elif c < len(reacts) and reacts[c] == -1:
                draw.text((tx + 6, badge_y), "DISLIKE", font=fontb, fill=RED)
            if c == neg_slot:
                draw.text((tx + (im.width if im else img_w) - 50, badge_y), "NEG", font=fontb, fill=RED)
            elif c == pos_slot:
                draw.text((tx + (im.width if im else img_w) - 46, badge_y), "POS", font=fontb, fill=BLUE)
            if c == widx:
                draw.text((tx + (im.width if im else img_w) - 24,
                           ty + cell["img_h"] - 26), "★", font=fontb, fill=GOLD)

            info_y = ty + cell["img_h"] + 8
            draw.text((x0 + CPAD, info_y), f"seed {seed}", font=font, fill=PURPLE)
            py = info_y + LINE_H
            for line in cell["plines"]:
                px_ = x0 + CPAD
                for piece, col in line:
                    draw.text((px_, py), piece, font=font, fill=col)
                    px_ += draw.textlength(piece, font=font)
                py += LINE_H

        if prev_winner_center is not None:
            pxc, _, pb = prev_winner_center
            for (cx_mid, ct, _) in centers:
                mid_y = (pb + ct) // 2
                draw.line([(pxc, pb), (pxc, mid_y)], fill=GOLD, width=2)
                draw.line([(pxc, mid_y), (cx_mid, mid_y)], fill=(90, 90, 100), width=2)
                draw.line([(cx_mid, mid_y), (cx_mid, ct)], fill=(90, 90, 100), width=2)

        prev_winner_center = centers[widx]
        y += row["row_h"] + ROW_GAP

    save_dir = get_game_dir(game_num)
    ts_name  = datetime.datetime.now().strftime("%H%M%S")
    path = os.path.join(save_dir, f"timeline_{ts_name}.png")
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Timeline UI hook
# ---------------------------------------------------------------------------

def build_lineage_html(lineage: list[dict], show_tests: bool = False) -> str:
    if not lineage:
        return "<p style='color:var(--text-muted);padding:1rem;font-size:14px;'>Winning prompts will appear here after each round…</p>"

    THUMB = 72
    rows_html = ""

    for entry in lineage:
        r            = entry["round"]
        winner_idx   = entry["winner_idx"]
        prompts      = entry["prompts"]
        thumb_uris   = entry["thumb_uris"]
        loras        = entry.get("loras", [])
        lora_w       = entry.get("lora_weight", 0.8)
        reactions    = entry.get("reactions", [])
        entry_neg_slot = entry.get("neg_test_slot")
        entry_pos_slot = entry.get("pos_test_slot")

        lora_pills = "".join(
            f"<span style='display:inline-block;margin:1px 2px;padding:1px 5px;"
            f"background:#1a2a3a;border:1px solid #3a6a9a;border-radius:4px;"
            f"font-size:9px;color:#7ab8e8;white-space:nowrap;'>"
            f"{name} {lora_w:.2f}</span>"
            for name in loras
        )
        lora_row = (
            f"<div style='margin:4px 0 2px;display:flex;flex-wrap:wrap;justify-content:center;gap:2px;'>{lora_pills}</div>"
            if loras else ""
        )

        cards = []
        for i, (uri, prompt) in enumerate(zip(thumb_uris, prompts)):
            is_winner = (i == winner_idx)
            reaction  = reactions[i] if i < len(reactions) else None
            label = LABELS[i] if i < len(LABELS) else str(i + 1)

            if is_winner:
                border = "2px solid #4a9eff"
            elif reaction == 1:
                border = "2px solid #50c050"
            elif reaction == -1:
                border = "2px solid #e05050"
            else:
                border = "1px solid #333"
            if is_winner:
                lbl_col = "#4a9eff"
            elif reaction == 1:
                lbl_col = "#50c050"
            elif reaction == -1:
                lbl_col = "#e05050"
            else:
                lbl_col = "#666"
            img_tag = (
                f"<img src='{uri}' style='width:{THUMB}px;height:auto;border-radius:4px;display:block;margin:0 auto;' />"
                if uri else
                f"<div style='width:{THUMB}px;height:{THUMB}px;background:#1a1a1a;border-radius:4px;margin:0 auto;'></div>"
            )
            if show_tests and i == entry_neg_slot:
                sticker = "<div style='font-size:7px;font-weight:700;color:#fff;background:#c0392b;border-radius:3px;padding:1px 3px;text-align:center;margin-top:1px;'>🧪 NEG</div>"
            elif show_tests and i == entry_pos_slot:
                sticker = "<div style='font-size:7px;font-weight:700;color:#fff;background:#27ae60;border-radius:3px;padding:1px 3px;text-align:center;margin-top:1px;'>🧪 POS</div>"
            else:
                sticker = ""
            winner_badge = "<div style='font-size:8px;color:#4a9eff;text-align:center;margin-top:2px;font-weight:600;'>WINNER</div>" if is_winner else ""
            cards.append(f"""
            <div style="flex:1 1 0;min-width:0;display:flex;flex-direction:column;align-items:center;
                        gap:3px;padding:5px;border:{border};border-radius:8px;
                        background:#111;box-sizing:border-box;overflow:hidden;">
              <div style="font-size:10px;font-weight:600;color:{lbl_col};">{label}</div>
              {img_tag}
              {sticker}
              {winner_badge}
              <div style="font-size:9px;color:#555;line-height:1.3;overflow-wrap:anywhere;
                          text-align:center;width:100%;margin-top:2px;">{prompt}</div>
            </div>""")

        rows_html += f"""
        <div style="width:100%;box-sizing:border-box;margin-bottom:10px;">
          <div style="font-size:10px;font-weight:500;color:#555;margin-bottom:5px;text-align:center;">Round {r}</div>
          {lora_row}
          <div style="display:flex;gap:4px;width:100%;box-sizing:border-box;">{"".join(cards)}</div>
        </div>"""

    return f"""<div style="padding:4px 0;font-family:var(--font-sans);width:100%;box-sizing:border-box;overflow:hidden;">{rows_html}</div>"""


# ---------------------------------------------------------------------------
# Leaderboard HTML
# ---------------------------------------------------------------------------

def build_leaderboard_html() -> str:
    top    = tag_scores.top_n(10)
    bottom = tag_scores.bottom_n(10)
    total  = len(tag_scores.get_scores())

    if total == 0:
        return "<p style='color:var(--text-muted);font-size:12px;padding:4px;'>No tag scores yet — play a round to start tracking.</p>"

    path_hint = tag_scores.scores_path()

    def fmt(score):
        if score == int(score):
            return f"{int(score):+d}"
        return f"{score:+.1f}"

    def colors(score):
        if score > 0:
            return "#1a3a1a", "#2a5a2a", "#50c050"
        if score < 0:
            return "#3a1a1a", "#5a2a2a", "#e05050"
        return "var(--surface-0)", "var(--surface-1)", "var(--text-muted)"

    rows_html = ""
    for i in range(max(len(bottom), len(top))):
        rank = i + 1
        if i < len(bottom):
            btag, bscore = bottom[i]
            pill_bg, badge_bg, txt = colors(bscore)
            left = f"""<div style="display:flex;align-items:center;gap:0;flex:1;min-width:0;justify-content:flex-end;">
              <div style="padding:5px 10px;display:flex;align-items:center;min-width:0;overflow:hidden;">
                <span style="font-size:13px;font-weight:500;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{btag}</span>
              </div>
              <div style="padding:5px 7px;background:{pill_bg};border-radius:6px 0 0 6px;display:flex;align-items:center;gap:4px;flex-shrink:0;">
                <span style="font-size:12px;font-weight:700;color:{txt};">{fmt(bscore)}</span>
                <div style="width:20px;height:20px;background:{badge_bg};border-radius:3px;display:flex;align-items:center;justify-content:center;">
                  <span style="font-size:10px;font-weight:700;color:{txt};">{rank}</span>
                </div>
              </div>
            </div>"""
        else:
            left = "<div style='flex:1;'></div>"

        if i < len(top):
            ttag, tscore = top[i]
            pill_bg, badge_bg, txt = colors(tscore)
            right = f"""<div style="display:flex;align-items:center;gap:0;flex:1;min-width:0;">
              <div style="padding:5px 7px;background:{pill_bg};border-radius:0 6px 6px 0;display:flex;align-items:center;gap:4px;flex-shrink:0;">
                <div style="width:20px;height:20px;background:{badge_bg};border-radius:3px;display:flex;align-items:center;justify-content:center;">
                  <span style="font-size:10px;font-weight:700;color:{txt};">{rank}</span>
                </div>
                <span style="font-size:12px;font-weight:700;color:{txt};">{fmt(tscore)}</span>
              </div>
              <div style="padding:5px 10px;display:flex;align-items:center;min-width:0;overflow:hidden;">
                <span style="font-size:13px;font-weight:500;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{ttag}</span>
              </div>
            </div>"""
        else:
            right = "<div style='flex:1;'></div>"

        rows_html += f"""<div style="display:flex;gap:8px;margin-bottom:4px;">{left}{right}</div>"""

    return f"""
    <div style="font-family:var(--font-sans);">
      <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;">{total:,} tags tracked · {path_hint}</div>
      <div style="display:flex;gap:8px;margin-bottom:6px;">
        <div style="flex:1;text-align:right;font-size:11px;font-weight:600;color:#e05050;text-transform:uppercase;letter-spacing:0.05em;">Bottom 10</div>
        <div style="flex:1;text-align:left;font-size:11px;font-weight:600;color:#50c050;text-transform:uppercase;letter-spacing:0.05em;">Top 10</div>
      </div>
      {rows_html}
    </div>"""


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_image(prompt: str, neg_prompt: str, steps: int, cfg: float,
                   width: int, height: int, seed: int = -1,
                   lora_selections: list[str] = None, lora_weight: float = 0.8):
    final_prompt = apply_loras_to_prompt(prompt, lora_selections or [], lora_weight)
    print(f"[prompt-evolution] Generating: {final_prompt}")
    from modules.processing import StableDiffusionProcessingTxt2Img, process_images
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        prompt=final_prompt,
        negative_prompt=neg_prompt,
        steps=steps,
        cfg_scale=cfg,
        seed=seed,
        width=width,
        height=height,
        sampler_name="Euler a",
        scheduler="Automatic",
        batch_size=1,
        n_iter=1,
        do_not_save_samples=True,
        do_not_save_grid=True,
    )
    p.scripts = scripts.scripts_txt2img
    p.script_args = ([None] * scripts.scripts_txt2img.constructor_args_count
                     if hasattr(scripts.scripts_txt2img, 'constructor_args_count') else [])
    processed = process_images(p)
    return processed.images[0] if processed.images else None


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

def initial_state():
    return {
        "round": 0,
        "prompts": [],          # list of N prompt strings
        "imgs": [],             # list of N PIL images
        "reactions": [],        # list of N reaction values (None/+1/-1)
        "winner_prompt": "",
        "lineage": [],
        "save_dir": None,
        "locked": [],
        "starting_tokens": [],
        "game": 1,
        "width": 512,
        "height": 512,
        "cfg": 7.0,
        "seed": -1,
        "lora_selections": [],
        "lora_weight": 0.8,
        "mutate_start": False,
        "tag_bias": 0,
        "n_choices": 2,
        "gen_counter": 0,       # total images generated this session
        "neg_test_slot": None,  # index of the current negativity test image (or None)
        "neg_test_tag": None,   # the tag injected for the negativity test
        "pos_test_slot": None,  # index of the current positivity test image (or None)
        "pos_test_tag": None,   # the tag injected for the positivity test
    }


# ---------------------------------------------------------------------------
# Mutation helpers for N choices
# ---------------------------------------------------------------------------

def random_prompt(locked: list[str], vocab: list[str], mutation_rate: int,
                  max_tags: int) -> str:
    """
    Build a fully random prompt for round 1. Starts from locked tokens only,
    then fills up to mutation_rate random tags from the extended vocab.
    No shared base — each call produces an independent result.
    """
    color_part_vocab = get_color_part_vocab()
    tokens = list(locked)  # start with locked/starting tokens only
    target = len(tokens) + mutation_rate
    target = min(target, max_tags)
    _scores = tag_scores.get_scores()
    _explore = tag_scores.get_setting("explore", 0.1)
    attempts = 0
    while len(tokens) < target and attempts < target * 10:
        attempts += 1
        tok = _pick_token(vocab, color_part_vocab, 0.0, [], [], scores=_scores,
                          explore=_explore)
        tok_l = tok.strip().lower()
        if tok_l in {t.strip().lower() for t in tokens}:
            continue
        if _would_exceed_cap(tok, tokens):
            continue
        tokens.append(tok)
    tokens = _deduplicate_tokens(tokens)
    tokens = _enforce_caps(tokens)
    return ", ".join(tokens)


_NEG_TEST_INTERVAL = 10  # every Nth image generated triggers a negativity test
_POS_TEST_OFFSET   = 1   # positivity test fires _NEG_TEST_INTERVAL + _POS_TEST_OFFSET gens later


def plan_negativity_test(state: dict, n: int, bias: float) -> tuple[int | None, str | None]:
    """
    Return (slot_idx, tag) if a negativity test should fire this round, else (None, None).
    Fires on generations that are multiples of _NEG_TEST_INTERVAL (10, 20, 30...).
    Only when bias > 0 and negative tags exist.
    """
    if bias <= 0:
        return None, None

    counter = state.get("gen_counter", 0)
    test_slot = None
    for i in range(n):
        if (counter + i + 1) % int(tag_scores.get_setting("test_interval", _NEG_TEST_INTERVAL)) == 0:
            test_slot = i
            break

    if test_slot is None:
        return None, None

    scores = tag_scores.get_scores()
    touches = tag_scores.get_touches()
    negative_tags = [t for t, s in scores.items() if s < 0]
    if not negative_tags:
        return None, None

    # Weight by |score| / (touches+1): extreme + uncertain tags get tested first
    weights = [abs(scores[t]) / (touches.get(t, 0) + 1) for t in negative_tags]
    return test_slot, random.choices(negative_tags, weights=weights, k=1)[0]


def plan_positivity_test(state: dict, n: int, bias: float) -> tuple[int | None, str | None]:
    """
    Return (slot_idx, tag) if a positivity test should fire this round, else (None, None).
    Fires on generations that are multiples of _NEG_TEST_INTERVAL + _POS_TEST_OFFSET (11, 21, 31...).
    Only when bias > 0 and positive tags exist.
    """
    if bias <= 0:
        return None, None

    counter = state.get("gen_counter", 0)
    test_slot = None
    for i in range(n):
        gen_num = counter + i + 1
        if gen_num % int(tag_scores.get_setting("test_interval", _NEG_TEST_INTERVAL)) == _POS_TEST_OFFSET:
            test_slot = i
            break

    if test_slot is None:
        return None, None

    scores = tag_scores.get_scores()
    touches = tag_scores.get_touches()
    positive_tags = [t for t, s in scores.items() if s > 0]
    if not positive_tags:
        return None, None

    weights = [abs(scores[t]) / (touches.get(t, 0) + 1) for t in positive_tags]
    return test_slot, random.choices(positive_tags, weights=weights, k=1)[0]


def inject_neg_test_tag(prompt: str, tag: str) -> str:
    """Append the negativity test tag to a prompt, avoiding duplicates."""
    tokens = [t.strip() for t in prompt.split(",") if t.strip()]
    if tag.lower() not in {t.lower() for t in tokens}:
        tokens.append(tag)
    return ", ".join(tokens)


def generate_n_prompts(base: str, vocab: list[str], n: int,
                       mutation_rate: int, locked: list[str], max_tags: int,
                       bias: float = 0.0) -> list[str]:
    """
    Generate N prompts by mutating base. Each gets a random number of
    mutations in [1, mutation_rate*2-1] averaging to mutation_rate.
    If bias > 0, mutations favour high-scoring tags and actively swap out low-scoring ones.
    """
    pool_tags, pool_weights, bad_tags = [], [], set()
    scores = tag_scores.get_scores()
    explore = tag_scores.get_setting("explore", 0.1)
    if bias > 0:
        k = tag_scores.get_meta().get("confidence_k", 5.0)
        pool_tags, pool_weights, bad_tags = tag_scores.weighted_pool(confidence_k=k)

    prompts = []
    seen = set()
    max_attempts = n * 10
    attempts = 0
    while len(prompts) < n and attempts < max_attempts:
        attempts += 1
        m = random.randint(1, max(1, mutation_rate * 2 - 1))
        candidate = mutate_prompt(base, vocab, m, locked, max_tags,
                                  bias=bias, pool_tags=pool_tags,
                                  pool_weights=pool_weights, bad_tags=bad_tags,
                                  scores=scores, explore=explore)
        key = frozenset(candidate.lower().split(", "))
        if key not in seen:
            seen.add(key)
            prompts.append(candidate)
    return prompts


# ---------------------------------------------------------------------------
# Pick button HTML builder — N dynamic buttons
# ---------------------------------------------------------------------------

def cards_update(imgs: list, prompts: list, width: int, height: int,
                  cfg: float, seed: int, generating_idx: int = -1,
                  show_tests: bool = False,
                  neg_slot: int | None = None,
                  pos_slot: int | None = None) -> list:
    """Return 9 gr.update(value=...) for card HTML slots."""
    updates = []
    for i in range(9):
        if i < len(prompts):
            if show_tests:
                test_type = "neg" if i == neg_slot else ("pos" if i == pos_slot else None)
            else:
                test_type = None
            html = build_image_card_html(
                imgs[i] if i < len(imgs) else None,
                prompts[i],
                width, height, cfg, seed,
                generating=(i == generating_idx),
                test_type=test_type,
            )
        else:
            html = ""
        updates.append(gr.update(value=html))
    return updates


def col_visibility(n: int) -> list:
    """Return 9 gr.update(visible=...) for card columns."""
    return [gr.update(visible=(i < n)) for i in range(9)]


def pick_buttons_update(n: int, visible: bool):
    """Return gr.update dicts for all 9 pick buttons."""
    updates = []
    for i in range(9):
        if visible and i < n:
            updates.append(gr.update(visible=True, value=f"✅ Pick {LABELS[i]}"))
        else:
            updates.append(gr.update(visible=False))
    return updates


def react_buttons_update(n: int, visible: bool, interactive: bool = True, variant: str = "secondary"):
    """Return gr.update dicts for all 18 react buttons (like+dislike × 9)."""
    updates = []
    for i in range(9):
        show = visible and i < n
        updates.append(gr.update(visible=show, interactive=interactive, variant=variant))
        updates.append(gr.update(visible=show, interactive=interactive, variant=variant))
    return updates


# ---------------------------------------------------------------------------
# Round logic — start_game
# ---------------------------------------------------------------------------

def start_game(player_prompt, neg_prompt, steps, cfg, width, height, mutation_rate, freq_min, freq_max,
               save_mode, fix_seed, seed_val, lora_selections, lora_weight, mutate_start, max_tags,
               tag_bias, n_choices, show_tests, state):
    n = int(n_choices)
    EMPTY_CARDS = cards_update([], [], width, height, cfg, -1)
    NO_COL  = col_visibility(n)
    NO_PICK = pick_buttons_update(n, False)
    NO_REACT = react_buttons_update(n, False)

    clear_stop()

    # Full reset — game number derived from disk so it persists across sessions/days
    game_num = resolve_game_num()
    state = initial_state()
    state["game"] = game_num

    vocab = None
    for status, progress, result in build_clip_vocab_with_progress(freq_min, freq_max):
        if is_stop_requested():
            yield (*EMPTY_CARDS, *NO_COL, *NO_PICK, *NO_REACT,
                   "⛔ Stopped.", state, gr.update(value="▶ Start Game"), gr.update(visible=False), build_lineage_html([]))
            return
        yield (*EMPTY_CARDS, *NO_COL, *NO_PICK, *NO_REACT,
               status, state, gr.update(interactive=False), gr.update(visible=True), build_lineage_html([]))
        if result is not None:
            vocab = result

    if not vocab:
        yield (*EMPTY_CARDS, *NO_COL, *NO_PICK, *NO_REACT,
               "⚠️ Could not build vocab — is a model loaded?",
               state, gr.update(interactive=True), gr.update(visible=False), build_lineage_html([]))
        return

    base = seed_prompt(player_prompt, vocab)
    starting_tokens = [t.strip() for t in player_prompt.split(",") if t.strip()]
    locked = [] if mutate_start else starting_tokens

    seed = int(seed_val) if fix_seed else -1
    lora_list = lora_selections if isinstance(lora_selections, list) else ([lora_selections] if lora_selections else [])

    bias = tag_bias / 100.0
    # Round 1: fully independent random prompts — no shared base, no bias, no duplicates
    prompts = []
    seen = set()
    max_attempts = n * 10
    attempts = 0
    while len(prompts) < n and attempts < max_attempts:
        attempts += 1
        candidate = random_prompt(starting_tokens, vocab, mutation_rate, max_tags)
        key = frozenset(candidate.lower().split(", "))
        if key not in seen:
            seen.add(key)
            prompts.append(candidate)
    imgs = [None] * n

    for i in range(n):
        if is_stop_requested():
            yield (*cards_update(imgs, prompts, width, height, cfg, seed),
                   *NO_COL, *NO_PICK, *NO_REACT,
                   "⛔ Stopped.", state,
                   gr.update(value="▶ Start Game", interactive=True),
                   gr.update(visible=False),
                   build_lineage_html([]))
            return
        cur_cards = cards_update(imgs, prompts, width, height, cfg, seed, generating_idx=i)
        yield (*cur_cards, *NO_COL, *NO_PICK, *NO_REACT,
               f"Round 1 · generating {LABELS[i]}…", state,
               gr.update(interactive=False),
               gr.update(visible=True),
               build_lineage_html([]))
        imgs[i] = generate_image(prompts[i], neg_prompt, steps, cfg, width, height,
                                  seed, lora_list, lora_weight)

    state["round"] = 1
    state["prompts"] = prompts
    state["imgs"] = imgs
    state["reactions"] = [None] * n
    state["locked"] = locked
    state["starting_tokens"] = starting_tokens
    state["width"] = width
    state["height"] = height
    state["cfg"] = cfg
    state["seed"] = seed
    state["lora_selections"] = lora_list
    state["lora_weight"] = lora_weight
    state["mutate_start"] = mutate_start
    state["tag_bias"] = tag_bias
    state["n_choices"] = n
    state["gen_counter"] = state.get("gen_counter", 0) + n
    state["neg_test_slot"] = None  # no negativity test on round 1
    state["neg_test_tag"]  = None
    state["pos_test_slot"] = None  # no positivity test on round 1
    state["pos_test_tag"]  = None

    ready_cards = cards_update(imgs, prompts, width, height, cfg, seed)
    yield (*ready_cards,
           *col_visibility(n),
           *pick_buttons_update(n, True),
           *react_buttons_update(n, True),
           "Round 1 · pick the best image",
           state,
           gr.update(value="🔄 Restart", interactive=True),
           gr.update(visible=False),
           build_lineage_html([]))


# ---------------------------------------------------------------------------
# Round logic — pick_winner
# ---------------------------------------------------------------------------

def pick_winner(winner_idx: int, neg_prompt, steps, cfg, width, height, mutation_rate,
                freq_min, freq_max, save_mode, fix_seed, seed_val, max_tags, show_tests, state):
    vocab    = get_clip_vocab(freq_min, freq_max)
    seed     = int(seed_val) if fix_seed else -1
    n        = state.get("n_choices", 2)
    lora_list   = state.get("lora_selections", [])
    lora_weight = state.get("lora_weight", 0.8)
    prompts  = state["prompts"]
    imgs     = state["imgs"]
    reactions = state.get("reactions", [None] * n)
    round_num = state["round"]
    game_num  = state.get("game", 1)

    winner_prompt = prompts[winner_idx]
    starting_set  = set(state.get("starting_tokens", []))
    prev_winner   = state.get("winner_prompt", "")
    prev_tags     = {t.strip().lower() for t in prev_winner.split(",") if t.strip()}

    # Commit buffered like/dislike reactions — only on tags NEW vs the parent
    # generation, with the reaction's weight split evenly across those new tags.
    LIKE_WEIGHT       = tag_scores.get_setting("like_weight", 1)
    DISLIKE_WEIGHT    = tag_scores.get_setting("dislike_weight", 3)
    DISLIKE_COLOR_W   = tag_scores.get_setting("dislike_color_weight", 1)
    for i, reaction in enumerate(reactions):
        if reaction is not None and i < len(prompts):
            new_tags = tag_scores.expand_new_tags(prompts[i], prev_winner,
                                                  exclude=starting_set)
            if new_tags:
                per = 1.0 / len(new_tags)   # dilute by how many tags changed
                if reaction > 0:
                    tag_scores.adjust_tags_set(new_tags, LIKE_WEIGHT * per)
                else:
                    colour_tags = {t for t in new_tags if t in _COLORS}
                    other_tags  = new_tags - colour_tags
                    if colour_tags:
                        tag_scores.adjust_tags_set(colour_tags, -DISLIKE_COLOR_W * per)
                    if other_tags:
                        tag_scores.adjust_tags_set(other_tags, -DISLIKE_WEIGHT * per)

    # Evaluate negativity test result
    neg_slot = state.get("neg_test_slot")
    neg_tag  = state.get("neg_test_tag")
    if neg_slot is not None and neg_tag is not None and neg_slot < len(reactions):
        neg_reaction = reactions[neg_slot]
        if neg_reaction == -1:
            # Explicitly disliked — strong confirmation it's bad
            tag_scores.adjust_tags_set({neg_tag}, -tag_scores.get_setting("test_strength", 10))
        elif neg_reaction == 1:
            # Explicitly liked — soften the penalty by halving current score
            scores = tag_scores.get_scores()
            current = scores.get(neg_tag, 0)
            if current < 0:
                tag_scores.adjust_tags_set({neg_tag}, -(current / 2))  # adds half back (halves magnitude)
        # ignored (None) — no action
    state["neg_test_slot"] = None
    state["neg_test_tag"]  = None

    # Evaluate positivity test result
    pos_slot = state.get("pos_test_slot")
    pos_tag  = state.get("pos_test_tag")
    if pos_slot is not None and pos_tag is not None and pos_slot < len(reactions):
        pos_reaction = reactions[pos_slot]
        if pos_reaction == 1:
            # Explicitly liked — strong confirmation it's good
            tag_scores.adjust_tags_set({pos_tag}, tag_scores.get_setting("test_strength", 10))
        elif pos_reaction == -1:
            # Explicitly disliked — soften the score by halving current score
            scores = tag_scores.get_scores()
            current = scores.get(pos_tag, 0)
            if current > 0:
                tag_scores.adjust_tags_set({pos_tag}, -(current / 2))  # halves magnitude
        # ignored (None) — no action
    state["pos_test_slot"] = None
    state["pos_test_tag"]  = None

    # Thumbnails for lineage
    thumb_uris = [pil_to_thumbnail_uri(img) for img in imgs]

    # Save
    save_msg = ""
    img_paths = [None] * len(imgs)   # per-slot saved file paths for the timeline
    if save_mode == "Save all":
        paths = save_images(imgs, [LABELS[i] for i in range(len(imgs))], game_num, round_num)
        if paths:
            state["save_dir"] = os.path.dirname(paths[0])
            save_msg = f" · {len(paths)} saved"
            img_paths = paths + [None] * (len(imgs) - len(paths))
    elif save_mode == "Save winner":
        paths = save_images([imgs[winner_idx]], [f"winner{LABELS[winner_idx]}"], game_num, round_num)
        if paths:
            state["save_dir"] = os.path.dirname(paths[0])
            save_msg = f" · winner ({LABELS[winner_idx]}) saved"
            img_paths[winner_idx] = paths[0]

    lineage_entry = {
        "round": round_num,
        "winner_idx": winner_idx,
        "prompts": list(prompts),
        "thumb_uris": thumb_uris,
        "img_paths": img_paths,   # saved file paths (None where not saved)
        "save_mode": save_mode,
        "loras": list(lora_list),
        "lora_weight": lora_weight,
        "reactions": list(reactions),
        "seed": seed,
        "neg_test_slot": neg_slot,
        "pos_test_slot": pos_slot,
    }
    state["lineage"].append(lineage_entry)
    state["winner_prompt"] = winner_prompt
    state["round"] += 1
    state["width"] = width
    state["height"] = height
    state["cfg"] = cfg
    state["seed"] = seed

    # Score the win: credit the tags the winner ADDED relative to the parent
    # generation, splitting +1 across them (1 new tag -> +1; k new -> +1/k each).
    tag_scores.record_win(winner_prompt, prev_winner, weight=1.0, exclude=starting_set)

    # Mild penalties (gentler than a dislike/neg-test), diluted across the tags
    # involved and colour-softened the same way dislikes are:
    #   - SWAPPED OUT: tags in the parent the winner dropped (won without them)
    #   - NOT PICKED:  mutations that showed up only in losing images
    SWAP_PEN   = tag_scores.get_swap_penalty()
    PICK_PEN   = tag_scores.get_pick_penalty()
    COLOR_SOFT = DISLIKE_COLOR_W / DISLIKE_WEIGHT  # colours penalised more softly

    def _mild_penalty(tags, base):
        tags = tags - starting_set
        if not tags or base <= 0:
            return
        per = base / len(tags)
        colour = {t for t in tags if t in _COLORS}
        other  = tags - colour
        if other:
            tag_scores.adjust_tags_set(other, -per)
        if colour:
            tag_scores.adjust_tags_set(colour, -per * COLOR_SOFT)

    # All expanded tags the winner actually has (so we never penalise a picked tag)
    winner_all = tag_scores.expand_new_tags(winner_prompt, "", exclude=starting_set)

    swapped_out = tag_scores.expand_new_tags(prev_winner, winner_prompt, exclude=starting_set)
    _mild_penalty(swapped_out, SWAP_PEN)

    test_slots = {s for s in (neg_slot, pos_slot) if s is not None}  # tests evaluated separately
    rejected = set()
    for i, p in enumerate(prompts):
        if i == winner_idx or i in test_slots:
            continue
        rejected |= tag_scores.expand_new_tags(p, prev_winner, exclude=starting_set)
    rejected -= winner_all
    _mild_penalty(rejected, PICK_PEN)

    # Close the round once: advance counter, apply decay, snapshot history
    tag_scores.end_round()

    next_round = state["round"]
    locked = [] if state.get("mutate_start") else state.get("locked", [])
    bias = state.get("tag_bias", 0) / 100.0
    new_prompts = generate_n_prompts(winner_prompt, vocab, n, mutation_rate, locked, max_tags, bias=bias)

    # Plan negativity test — isolate the tag: parent prompt + ONLY the test tag
    neg_slot, neg_tag = plan_negativity_test(state, n, bias)
    if neg_slot is not None and neg_tag is not None:
        new_prompts[neg_slot] = inject_neg_test_tag(winner_prompt, neg_tag)
    state["neg_test_slot"] = neg_slot
    state["neg_test_tag"]  = neg_tag

    # Plan positivity test — offset by 1 from negativity test, never same slot
    pos_slot, pos_tag = plan_positivity_test(state, n, bias)
    if pos_slot is not None and pos_slot == neg_slot:
        pos_slot = None  # safety: never overlap (shouldn't happen by design)
        pos_tag  = None
    if pos_slot is not None and pos_tag is not None:
        new_prompts[pos_slot] = inject_neg_test_tag(winner_prompt, pos_tag)
    state["pos_test_slot"] = pos_slot
    state["pos_test_tag"]  = pos_tag

    state["gen_counter"] = state.get("gen_counter", 0) + n
    new_imgs = [None] * n

    lineage_html     = build_lineage_html(state["lineage"], show_tests=show_tests)
    leaderboard_html = build_leaderboard_html()
    NO_COL   = col_visibility(n)
    NO_PICK  = pick_buttons_update(n, False)
    NO_REACT = react_buttons_update(n, False)

    for i in range(n):
        if is_stop_requested():
            state["prompts"]   = new_prompts
            state["imgs"]      = new_imgs
            state["reactions"] = [None] * n
            yield (*cards_update(new_imgs, new_prompts, width, height, cfg, seed),
                   *NO_COL, *NO_PICK, *NO_REACT,
                   state, lineage_html, leaderboard_html,
                   "⛔ Stopped.")
            return
        cur_cards = cards_update(new_imgs, new_prompts, width, height, cfg, seed,
                                 generating_idx=i, show_tests=show_tests,
                                 neg_slot=neg_slot, pos_slot=pos_slot)
        yield (*cur_cards, *NO_COL, *NO_PICK, *NO_REACT,
               state, lineage_html, leaderboard_html,
               f"Round {next_round} · generating {LABELS[i]}…")
        new_imgs[i] = generate_image(new_prompts[i], neg_prompt, steps, cfg, width, height,
                                      seed, lora_list, lora_weight)

    state["prompts"]   = new_prompts
    state["imgs"]      = new_imgs
    state["reactions"] = [None] * n

    ready_cards = cards_update(new_imgs, new_prompts, width, height, cfg, seed,
                               show_tests=show_tests, neg_slot=neg_slot, pos_slot=pos_slot)
    yield (*ready_cards,
           *col_visibility(n),
           *pick_buttons_update(n, True),
           *react_buttons_update(n, True),
           state, lineage_html, leaderboard_html,
           f"Round {next_round} · pick the best image{save_msg}")


# ---------------------------------------------------------------------------
# React (like/dislike) handler
# ---------------------------------------------------------------------------

def reroll_round(neg_prompt, steps, cfg, width, height, mutation_rate,
                 freq_min, freq_max, fix_seed, seed_val, max_tags, show_tests, state):
    """
    Regenerate the current round's mutations from the same parent, without
    scoring anything. Disabled when the seed is fixed (identical results).
    """
    n = state.get("n_choices", 2)
    prev_winner = state.get("winner_prompt", "")
    cur = state.get("prompts", [])
    if not cur or not prev_winner:
        yield (*cards_update(state.get("imgs", []), cur, width, height, cfg, -1),
               *col_visibility(n), *pick_buttons_update(n, bool(cur)),
               *react_buttons_update(n, bool(cur)),
               state, build_lineage_html(state.get("lineage", []), show_tests=show_tests),
               build_leaderboard_html(), "Nothing to reroll — pick a winner first.")
        return
    if fix_seed:
        yield (*cards_update(state.get("imgs", []), cur, width, height, cfg, int(seed_val)),
               *col_visibility(n), *pick_buttons_update(n, True),
               *react_buttons_update(n, True),
               state, build_lineage_html(state.get("lineage", []), show_tests=show_tests),
               build_leaderboard_html(), "🎲 Reroll is disabled while the seed is fixed.")
        return

    vocab = get_clip_vocab(freq_min, freq_max)
    seed  = -1
    lora_list   = state.get("lora_selections", [])
    lora_weight = state.get("lora_weight", 0.8)
    locked = [] if state.get("mutate_start") else state.get("locked", [])
    bias   = state.get("tag_bias", 0) / 100.0

    new_prompts = generate_n_prompts(prev_winner, vocab, n, mutation_rate, locked,
                                     max_tags, bias=bias)
    # No tests on a reroll
    state["neg_test_slot"] = None
    state["neg_test_tag"]  = None
    state["pos_test_slot"] = None
    state["pos_test_tag"]  = None
    state["gen_counter"] = state.get("gen_counter", 0) + n

    new_imgs = [None] * n
    lineage_html     = build_lineage_html(state["lineage"], show_tests=show_tests)
    leaderboard_html = build_leaderboard_html()
    NO_COL, NO_PICK, NO_REACT = (col_visibility(n), pick_buttons_update(n, False),
                                 react_buttons_update(n, False))

    for i in range(n):
        if is_stop_requested():
            break
        yield (*cards_update(new_imgs, new_prompts, width, height, cfg, seed,
                             generating_idx=i),
               *NO_COL, *NO_PICK, *NO_REACT,
               state, lineage_html, leaderboard_html,
               f"🎲 Rerolling · generating {LABELS[i]}…")
        new_imgs[i] = generate_image(new_prompts[i], neg_prompt, steps, cfg, width,
                                     height, seed, lora_list, lora_weight)

    state["prompts"]   = new_prompts
    state["imgs"]      = new_imgs
    state["reactions"] = [None] * n
    yield (*cards_update(new_imgs, new_prompts, width, height, cfg, seed),
           *col_visibility(n), *pick_buttons_update(n, True),
           *react_buttons_update(n, True),
           state, lineage_html, leaderboard_html,
           "🎲 Rerolled — pick the best image")


def react_image(idx: int, delta: int, state):
    reactions = state.get("reactions", [])
    prompts   = state.get("prompts", [])
    if idx >= len(prompts) or not prompts[idx]:
        return (state,
                gr.update(), gr.update(),
                "Nothing to rate yet — start a game first.")

    current = reactions[idx] if idx < len(reactions) else None

    # Toggle off
    if current == delta:
        reactions[idx] = None
        state["reactions"] = reactions
        return (state,
                gr.update(interactive=True, variant="secondary"),
                gr.update(interactive=True, variant="secondary"),
                f"Image {LABELS[idx]} rating cleared")

    reactions[idx] = delta
    state["reactions"] = reactions
    verb = "liked" if delta > 0 else "disliked"
    like_v    = "primary"   if delta > 0 else "secondary"
    dislike_v = "secondary" if delta > 0 else "primary"
    return (state,
            gr.update(interactive=True, variant=like_v),
            gr.update(interactive=True, variant=dislike_v),
            f"Image {LABELS[idx]} {verb} · applies when you pick")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_TABLE_CAP = 50  # rows shown per end (top + bottom) when not searching


def _scores_table_df(search: str = "", cap: int = _TABLE_CAP):
    """
    Editable table: tag, score, touches.
    - With a search term: every tag whose name contains it (no cap).
    - Otherwise: top `cap` + bottom `cap` tags only, so huge dicts stay snappy.
    """
    import pandas as pd
    scores = tag_scores.get_scores()
    touches = tag_scores.get_touches()
    cols = ["tag", "score", "touches"]
    if not scores:
        return pd.DataFrame(columns=cols)

    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    search = (search or "").strip().lower()
    if search:
        items = [(t, s) for t, s in items if search in t]
    elif len(items) > 2 * cap:
        items = items[:cap] + items[-cap:]  # already sorted desc → top then bottom

    rows = [(t, s, touches.get(t, 0)) for t, s in items]
    return pd.DataFrame(rows, columns=cols)



def _tag_choices():
    return sorted(tag_scores.get_scores().keys())


def _extreme_tags(n: int = 10):
    """Top n + bottom n tag names (deduped, no overlap when the dict is small)."""
    seen, out = set(), []
    for t, _ in tag_scores.top_n(n) + tag_scores.bottom_n(n):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out



def _scores_bar_html(top: int = 15) -> str:
    """
    Render a horizontal SVG bar chart of the top N and bottom N tags.
    Returns raw HTML; works in Gradio 3 and 4.
    """
    scores = tag_scores.get_scores()
    touches = tag_scores.get_touches()
    if not scores:
        return "<p style='color:#888;padding:1em'>No scores yet — play some rounds first.</p>"

    items = sorted(scores.items(), key=lambda x: x[1])
    picked, seen = [], set()
    for t, s in items[:top] + items[-top:]:
        if t not in seen:
            seen.add(t)
            picked.append((t, round(s, 3)))
    picked.sort(key=lambda x: x[1])   # lowest → highest (chart reads bottom-up)

    row_h   = 24
    label_w = 160
    bar_area = 500
    pad     = 12
    n       = len(picked)
    height  = n * row_h + pad * 2

    max_abs = max(abs(s) for _, s in picked) or 1
    zero_x  = label_w + int(bar_area * (max_abs / (2 * max_abs)))   # midpoint

    rows = []
    for i, (tag, score) in enumerate(picked):
        y     = pad + i * row_h
        cy    = y + row_h // 2
        color = "#4caf50" if score > 0 else ("#f44336" if score < 0 else "#888")
        bar_w = int(abs(score) / max_abs * (bar_area // 2))
        x0    = zero_x - bar_w if score < 0 else zero_x
        # label
        rows.append(
            f'<text x="{label_w - 6}" y="{cy + 5}" '
            f'text-anchor="end" font-size="12" fill="#ddd">{tag}</text>'
        )
        # bar
        if bar_w > 0:
            rows.append(
                f'<rect x="{x0}" y="{y + 3}" width="{bar_w}" height="{row_h - 6}" '
                f'fill="{color}" rx="3"/>'
            )
        # score label
        tx = x0 + bar_w + 4 if score >= 0 else x0 - 4
        anchor = "start" if score >= 0 else "end"
        rows.append(
            f'<text x="{tx}" y="{cy + 5}" text-anchor="{anchor}" '
            f'font-size="11" fill="#aaa">{score:+.2f}'
            f'<tspan fill="#666"> ×{touches.get(tag, 0)}</tspan></text>'
        )

    # zero line
    zero_line = (
        f'<line x1="{zero_x}" y1="{pad}" x2="{zero_x}" y2="{height - pad}" '
        f'stroke="#555" stroke-width="1"/>'
    )

    svg_w = label_w + bar_area + 60
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="100%" viewBox="0 0 {svg_w} {height}" '
        f'style="background:#1a1a1a;border-radius:8px;display:block">'
        f'{zero_line}{"".join(rows)}'
        f'</svg>'
    )
    return svg


def _trend_html(tags) -> str:
    """
    SVG line chart of score-over-rounds for the given tags.
    Each tag gets its own colour; axes auto-scale to the data.
    """
    if not tags:
        return "<p style='color:#888;padding:1em'>Select tags above and hit Plot.</p>"

    hist = tag_scores.get_history(list(tags))
    # Flatten to check we actually have data
    all_pts = [(r, s, t) for t in tags for r, s in hist.get(t, [])]
    if not all_pts:
        return "<p style='color:#888;padding:1em'>No history yet — scores are recorded round by round as you play.</p>"

    W, H   = 900, 360
    PAD_L  = 52   # room for y-axis labels
    PAD_R  = 16
    PAD_T  = 28
    PAD_B  = 36   # room for x-axis labels

    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    min_r  = min(r for r, s, _ in all_pts)
    max_r  = max(r for r, s, _ in all_pts)
    min_s  = min(s for _, s, _ in all_pts)
    max_s  = max(s for _, s, _ in all_pts)
    rng_r  = max_r - min_r or 1
    rng_s  = max_s - min_s or 1

    def px(r, s):
        x = PAD_L + (r - min_r) / rng_r * plot_w
        y = PAD_T + (1 - (s - min_s) / rng_s) * plot_h
        return x, y

    # 20 distinct colours (enough for up to top50+bottom50 if needed)
    COLORS = [
        "#4fc3f7","#81c784","#e57373","#ffb74d","#ce93d8",
        "#80cbc4","#fff176","#f48fb1","#a5d6a7","#90caf9",
        "#ffcc80","#b39ddb","#80deea","#ef9a9a","#c5e1a5",
        "#ffe082","#bcaaa4","#b0bec5","#ff8a65","#4db6ac",
    ]

    elems = []

    # zero line
    if min_s <= 0 <= max_s:
        _, zy = px(min_r, 0)
        elems.append(
            f'<line x1="{PAD_L}" y1="{zy:.1f}" x2="{W - PAD_R}" y2="{zy:.1f}" '
            f'stroke="#444" stroke-width="1" stroke-dasharray="4,3"/>'
        )

    # y-axis grid + labels (5 ticks)
    for i in range(5):
        s_val = min_s + rng_s * i / 4
        _, y  = px(min_r, s_val)
        elems.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" '
            f'stroke="#2a2a2a" stroke-width="1"/>'
        )
        elems.append(
            f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#777">{s_val:+.1f}</text>'
        )

    # x-axis ticks (up to 8)
    n_xticks = min(8, int(rng_r) + 1)
    for i in range(n_xticks):
        r_val = min_r + rng_r * i / max(n_xticks - 1, 1)
        x, _  = px(r_val, min_s)
        elems.append(
            f'<text x="{x:.1f}" y="{H - PAD_B + 16}" text-anchor="middle" '
            f'font-size="11" fill="#777">r{int(round(r_val))}</text>'
        )

    # Lines + dots per tag
    for ti, tag in enumerate(tags):
        pts = sorted(hist.get(tag, []), key=lambda p: p[0])
        if not pts:
            continue
        col = COLORS[ti % len(COLORS)]
        coords = [px(r, s) for r, s in pts]
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
            for i, (x, y) in enumerate(coords)
        )
        elems.append(
            f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for (x, y), (r, s) in zip(coords, pts):
            elems.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}">'
                f'<title>{tag} · r{r} · {s:+.3f}</title></circle>'
            )

    # Legend (right-aligned inside chart)
    for ti, tag in enumerate(tags):
        if not hist.get(tag):
            continue
        col = COLORS[ti % len(COLORS)]
        lx  = W - PAD_R - 8
        ly  = PAD_T + 14 + ti * 18
        elems.append(f'<rect x="{lx - 28}" y="{ly - 10}" width="16" height="3" fill="{col}"/>')
        elems.append(
            f'<text x="{lx - 8}" y="{ly}" text-anchor="end" '
            f'font-size="11" fill="{col}">{tag}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" '
        f'viewBox="0 0 {W} {H}" '
        f'style="background:#1a1a1a;border-radius:8px;display:block">'
        + "".join(elems) +
        f'</svg>'
    )
    return svg


def build_scores_tab():
    """Self-contained 📊 Tag Scores tab: bar chart + manual editing."""
    with gr.Blocks(analytics_enabled=False) as scores_ui:
        gr.Markdown("## 📊 Tag Scores")
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
            wipe_btn = gr.Button("🗑️ Wipe all scores", size="sm", variant="stop")
        status = gr.Markdown("")

        plot = gr.HTML(value=_scores_bar_html(), label="Top & bottom tags")

        with gr.Accordion("⚙️ Pool tuning", open=False):
            gr.Markdown(
                "**Decay** multiplies every score each round (1.0 = off; "
                "0.98 fades stale preferences). **Confidence k** damps "
                "rarely-seen tags in the mutation pool: weight × touches/(touches+k); "
                "set 0 to use raw scores. **Swap-out** / **not-picked** penalties are "
                "mild negatives (split across the tags involved, 0 = off) for tags the "
                "winner dropped and for losing mutations.")
            with gr.Row():
                decay_slider = gr.Slider(0.90, 1.00, value=tag_scores.get_decay(),
                                         step=0.005, label="Score decay / round")
                conf_slider = gr.Slider(0, 25, value=tag_scores.get_confidence_k(),
                                        step=1, label="Confidence k")
            with gr.Row():
                swap_slider = gr.Slider(0, 3, value=tag_scores.get_swap_penalty(),
                                        step=0.25, label="Swap-out penalty")
                pick_slider = gr.Slider(0, 3, value=tag_scores.get_pick_penalty(),
                                        step=0.25, label="Not-picked penalty")
            with gr.Row():
                test_str_slider = gr.Slider(1, 20, value=tag_scores.get_setting("test_strength", 10),
                                            step=1, label="Test strength (± on confirm)")
                test_int_slider = gr.Slider(5, 30, value=tag_scores.get_setting("test_interval", 10),
                                            step=1, label="Test interval (gens)")
            with gr.Row():
                like_slider = gr.Slider(0, 3, value=tag_scores.get_setting("like_weight", 1),
                                        step=0.5, label="Like weight")
                dislike_slider = gr.Slider(0, 6, value=tag_scores.get_setting("dislike_weight", 3),
                                           step=0.5, label="Dislike weight")
                explore_slider = gr.Slider(0, 0.5, value=tag_scores.get_setting("explore", 0.1),
                                           step=0.05, label="Exploration bonus")

        gr.Markdown("### 📈 Score trends over rounds")
        with gr.Row():
            trend_n = gr.Slider(5, 50, value=10, step=5,
                                label="Top/bottom N", scale=2)
            auto_trend_btn = gr.Button("⭐ Auto top/bottom N", scale=1)
        with gr.Row():
            trend_pick = gr.Dropdown(
                choices=_tag_choices(),
                value=_extreme_tags(10),
                label="Tags to plot", multiselect=True, scale=3)
            trend_btn = gr.Button("Plot selected", scale=1)
        trend_plot = gr.HTML(value=_trend_html(_extreme_tags(10)))

        gr.Markdown("### ✏️ Adjust a tag")
        with gr.Row():
            tag_pick = gr.Dropdown(choices=_tag_choices(), label="Tag",
                                   allow_custom_value=True)
            minus_btn = gr.Button("➖ −1")
            plus_btn = gr.Button("➕ +1")
        with gr.Row():
            set_val = gr.Number(label="Set exact score", value=0)
            set_btn = gr.Button("Set", variant="primary")
            del_btn = gr.Button("Delete tag", variant="stop")

        gr.Markdown(f"### 📋 Edit the table directly, then Apply "
                    f"(delete a row to remove a tag). Shows top/bottom {_TABLE_CAP} "
                    f"unless you search.")
        with gr.Row():
            search_box = gr.Textbox(label="🔍 Filter tags", placeholder="e.g. eyes",
                                    scale=4)
            prune_btn = gr.Button("🧹 Remove zero-score tags", scale=2)
        table = gr.Dataframe(
            value=_scores_table_df(),
            headers=["tag", "score", "touches"],
            datatype=["str", "number", "number"],
            interactive=True, wrap=True,
        )
        apply_btn = gr.Button("💾 Apply table edits", variant="primary")

        # ── handlers ──────────────────────────────────────────────────────
        def _records(df):
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                return df.to_dict("records")
            return [{"tag": r[0], "score": r[1] if len(r) > 1 else 0}
                    for r in (df or [])]

        def _refresh(search):
            return (_scores_bar_html(), _scores_table_df(search),
                    gr.update(choices=_tag_choices()), "")

        def _nudge(tag, d, search):
            if tag and tag.strip():
                tag_scores.bump_tag(tag, d)
                msg = f"`{tag.strip().lower()}` {d:+g}"
            else:
                msg = "Pick a tag first."
            return _scores_bar_html(), _scores_table_df(search), msg

        def _set(tag, val, search):
            if tag and tag.strip():
                tag_scores.set_score(tag, val)
                msg = f"Set `{tag.strip().lower()}` = {val}"
            else:
                msg = "Pick a tag first."
            return (_scores_bar_html(), _scores_table_df(search),
                    gr.update(choices=_tag_choices()), msg)

        def _del(tag, search):
            if tag and tag.strip():
                tag_scores.delete_tag(tag)
                msg = f"Deleted `{tag.strip().lower()}`"
            else:
                msg = "Pick a tag first."
            return (_scores_bar_html(), _scores_table_df(search),
                    gr.update(choices=_tag_choices(), value=None), msg)

        def _apply(df, search):
            # The table only shows a subset, so MERGE: update the rows present and
            # delete only tags that were shown before but removed from the table.
            shown_before = set(_scores_table_df(search)["tag"].tolist())
            mapping = {}
            for r in _records(df):
                t = str(r.get("tag", "")).strip().lower()
                if not t:
                    continue
                try:
                    mapping[t] = float(r.get("score"))
                except (TypeError, ValueError):
                    continue
            tag_scores.set_many(mapping, replace=False)
            removed = shown_before - set(mapping.keys())
            for t in removed:
                tag_scores.delete_tag(t)
            return (_scores_bar_html(), _scores_table_df(search),
                    gr.update(choices=_tag_choices()),
                    f"Updated {len(mapping)}, removed {len(removed)}.")

        def _prune(search):
            n = tag_scores.prune_zeros()
            return (_scores_bar_html(), _scores_table_df(search),
                    gr.update(choices=_tag_choices()),
                    f"Pruned {n} zero-score tag(s).")

        def _wipe():
            tag_scores.clear_scores()
            return (_scores_bar_html(), _scores_table_df(),
                    gr.update(choices=[], value=None), "Wiped all scores.")

        def _set_decay(v):
            tag_scores.set_decay(v)
            return f"Decay set to {v:.3f} / round."

        def _set_conf(v):
            tag_scores.set_confidence_k(v)
            return f"Confidence k set to {v:g}."

        def _set_swap(v):
            tag_scores.set_swap_penalty(v)
            return f"Swap-out penalty set to {v:g}."

        def _set_pick(v):
            tag_scores.set_pick_penalty(v)
            return f"Not-picked penalty set to {v:g}."

        def _plot_trends(tags):
            return _trend_html(tags or [])

        def _auto_trends(n):
            tags = _extreme_tags(int(n))
            return gr.update(choices=_tag_choices(), value=tags), _trend_html(tags)

        refresh_btn.click(_refresh, inputs=[search_box],
                          outputs=[plot, table, tag_pick, status])
        refresh_btn.click(_auto_trends, inputs=[trend_n],
                          outputs=[trend_pick, trend_plot])
        search_box.change(lambda s: _scores_table_df(s), inputs=[search_box],
                          outputs=[table])
        plus_btn.click(lambda t, s: _nudge(t, +1, s), inputs=[tag_pick, search_box],
                       outputs=[plot, table, status])
        minus_btn.click(lambda t, s: _nudge(t, -1, s), inputs=[tag_pick, search_box],
                        outputs=[plot, table, status])
        set_btn.click(_set, inputs=[tag_pick, set_val, search_box],
                      outputs=[plot, table, tag_pick, status])
        del_btn.click(_del, inputs=[tag_pick, search_box],
                      outputs=[plot, table, tag_pick, status])
        apply_btn.click(_apply, inputs=[table, search_box],
                        outputs=[plot, table, tag_pick, status])
        prune_btn.click(_prune, inputs=[search_box],
                        outputs=[plot, table, tag_pick, status])
        wipe_btn.click(_wipe, outputs=[plot, table, tag_pick, status])
        decay_slider.release(_set_decay, inputs=[decay_slider], outputs=[status])
        conf_slider.release(_set_conf, inputs=[conf_slider], outputs=[status])
        swap_slider.release(_set_swap, inputs=[swap_slider], outputs=[status])
        pick_slider.release(_set_pick, inputs=[pick_slider], outputs=[status])

        def _mk_setting(name, label):
            def f(v):
                tag_scores.set_setting(name, v)
                return f"{label} set to {v:g}."
            return f
        test_str_slider.release(_mk_setting("test_strength", "Test strength"),
                                inputs=[test_str_slider], outputs=[status])
        test_int_slider.release(_mk_setting("test_interval", "Test interval"),
                                inputs=[test_int_slider], outputs=[status])
        like_slider.release(_mk_setting("like_weight", "Like weight"),
                            inputs=[like_slider], outputs=[status])
        dislike_slider.release(_mk_setting("dislike_weight", "Dislike weight"),
                               inputs=[dislike_slider], outputs=[status])
        explore_slider.release(_mk_setting("explore", "Exploration bonus"),
                               inputs=[explore_slider], outputs=[status])
        trend_btn.click(_plot_trends, inputs=[trend_pick], outputs=[trend_plot])
        auto_trend_btn.click(_auto_trends, inputs=[trend_n],
                             outputs=[trend_pick, trend_plot])
        trend_n.release(_auto_trends, inputs=[trend_n],
                        outputs=[trend_pick, trend_plot])

    return scores_ui


def on_ui_tabs():
    css = """
    #pe-layout { display: flex !important; flex-direction: row !important; gap: 16px; align-items: flex-start; }
    #pe-left { flex: 0 0 60%; min-width: 0; }
    #pe-right { flex: 1; min-width: 0; overflow-y: auto; box-sizing: border-box; }
    #pe-bottom-layout { display: flex !important; flex-direction: row !important; gap: 16px; align-items: flex-start; margin-top: 16px; width: 100%; box-sizing: border-box; }
    #pe-bottom-left { flex: 0 0 60%; min-width: 0; overflow-x: hidden; box-sizing: border-box; }
    #pe-bottom-right { flex: 1 1 0; min-width: 0; box-sizing: border-box; overflow-x: hidden; }
    #pe-status textarea { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #pe-lineage { width: 100%; box-sizing: border-box; overflow-x: hidden; }
    #pe-bottom-left .block, #pe-bottom-right .block, #pe-lineage, #pe-leaderboard { border: none !important; background: transparent !important; }
    """

    with gr.Blocks(analytics_enabled=False) as ui:
        gr.HTML(f"<style>{css}</style>")
        game_state = gr.State(initial_state())

        with gr.Row(elem_id="pe-layout"):

            # ── Left: 9 image columns, each with card HTML + pick + like/dislike ──
            with gr.Column(elem_id="pe-left"):
                card_htmls = []   # 9 gr.HTML image cards
                pick_btns  = []   # 9 pick buttons
                react_btns = []   # 18 react buttons (like, dislike × 9)
                card_cols  = []   # 9 gr.Column wrappers (for show/hide)

                for row in range(3):
                    with gr.Row():
                        for col_idx in range(3):
                            i = row * 3 + col_idx
                            with gr.Column(visible=(i < 2), min_width=80) as col:
                                card_cols.append(col)
                                card_htmls.append(gr.HTML(value="", label=LABELS[i]))
                                pick_btns.append(gr.Button(f"✅ Pick {LABELS[i]}", visible=False, size="sm"))
                                like_btn    = gr.Button("👍", visible=False, size="sm")
                                dislike_btn = gr.Button("👎", visible=False, size="sm")
                                react_btns.append(like_btn)
                                react_btns.append(dislike_btn)

            # ── Right: controls ──────────────────────────────────────────────
            with gr.Column(elem_id="pe-right"):
                gr.Markdown("## The A/B Game")

                start_btn   = gr.Button("▶ Start Game", variant="primary")
                stop_btn    = gr.Button("⛔ Stop", variant="stop", visible=False)
                reroll_btn  = gr.Button("🎲 Reroll round", visible=False)
                status_text = gr.Textbox(label="Status", interactive=False,
                                         elem_id="pe-status", max_lines=1)

                with gr.Accordion("⚙️ Settings", open=True):
                    open_folder_btn = gr.Button("📂 Open output folder")

                    save_mode = gr.Radio(
                        choices=["Save all", "Save winner", "Don't save"],
                        value="Save all", label="💾 Save images",
                    )
                    n_choices = gr.Slider(2, 9, value=2, step=1, label="Number of choices")
                    player_prompt = gr.Textbox(
                        label="Starting prompt (leave blank for random)",
                        placeholder="e.g. ancient temple, misty forest …",
                    )
                    mutate_start = gr.Checkbox(
                        label="Mutate starting prompt (tokens can be replaced or removed)",
                        value=False,
                    )
                    neg_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="ugly, blurry, watermark …",
                    )
                    steps         = gr.Slider(1, 50, value=20, step=1, label="Steps")
                    cfg           = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFG scale")
                    with gr.Row():
                        width  = gr.Slider(256, 2048, value=512, step=64, label="Width")
                        height = gr.Slider(256, 2048, value=512, step=64, label="Height")
                    mutation_rate = gr.Slider(1, 10, value=3, step=1, label="Mutations per image (avg)")
                    max_tags      = gr.Slider(5, 75, value=20, step=1,
                                              label="Max tags (soft cap — scales swap vs add chance)")
                    tag_bias      = gr.Slider(0, 100, value=0, step=5,
                                              label="Tag bias % (0 = random, 100 = always pull a top-scoring tag)")
                    with gr.Row():
                        fix_seed   = gr.Checkbox(label="Fix seed", value=False)
                        seed_input = gr.Number(label="Seed", value=42, precision=0, visible=False)
                    fix_seed.change(lambda x: gr.update(visible=x), inputs=[fix_seed], outputs=[seed_input])
                    show_tests = gr.Checkbox(label="Show negativity/positivity test markers", value=False)
                    with gr.Row():
                        freq_min = gr.Slider(1, 8, value=5, step=0.5, label="Min post count (log)")
                        freq_max = gr.Slider(1, 8, value=8, step=0.5, label="Max post count (log)")

                    with gr.Accordion("🎨 LoRA", open=False):
                        lora_refresh_btn = gr.Button("🔄 Refresh LoRA list", size="sm")
                        available_loras  = get_available_loras()
                        lora_selector    = gr.Dropdown(
                            choices=available_loras, value=[], multiselect=True,
                            label="Active LoRAs (applied every generation, not mutated)",
                            info=f"{len(available_loras)} LoRA(s) found" if available_loras else "No LoRAs found",
                        )
                        lora_weight = gr.Slider(0.0, 2.0, value=0.8, step=0.05,
                                                label="LoRA weight")

                        def refresh_loras():
                            loras = get_available_loras()
                            return gr.update(choices=loras, value=[],
                                             info=f"{len(loras)} LoRA(s) found" if loras else "No LoRAs found")
                        lora_refresh_btn.click(refresh_loras, outputs=[lora_selector])

        # ── Below: lineage + leaderboard ─────────────────────────────────────
        with gr.Row(elem_id="pe-bottom-layout"):
            with gr.Column(elem_id="pe-bottom-left"):
                with gr.Row():
                    gr.HTML("<div style='font-size:13px;font-weight:500;color:var(--text-secondary);margin-bottom:6px;'>📜 Winning prompt lineage</div>")
                    timeline_btn = gr.Button("🧬 Generate Timeline", size="sm")
                lineage_display = gr.HTML(value=build_lineage_html([]), elem_id="pe-lineage")
                timeline_img = gr.Image(label="Timeline", visible=False, interactive=False)
            with gr.Column(elem_id="pe-bottom-right"):
                with gr.Row():
                    gr.HTML("<div style='font-size:13px;font-weight:500;color:var(--text-secondary);margin-bottom:6px;'>🏆 Tag leaderboard</div>")
                    wipe_scores_btn = gr.Button("🗑️ Wipe scores", size="sm", variant="stop")
                leaderboard_display = gr.HTML(value=build_leaderboard_html(), elem_id="pe-leaderboard")

        # ── Wiring ───────────────────────────────────────────────────────────

        def open_output_folder(state):
            save_dir = state.get("save_dir") or get_save_dir()
            try:
                os.startfile(save_dir)
            except AttributeError:
                import subprocess
                try:
                    subprocess.Popen(["xdg-open", save_dir])
                except Exception:
                    subprocess.Popen(["open", save_dir])
            return f"Opened: {save_dir}"

        open_folder_btn.click(open_output_folder, inputs=[game_state], outputs=[status_text])

        # Outputs: 9 card HTMLs + 9 col visibility + 9 pick btns + 18 react btns + ...
        ALL_OUTPUTS  = [*card_htmls, *card_cols, *pick_btns, *react_btns,
                        status_text, game_state, start_btn, stop_btn, lineage_display]
        PICK_OUTPUTS = [*card_htmls, *card_cols, *pick_btns, *react_btns,
                        game_state, lineage_display, leaderboard_display, status_text]

        start_btn.click(
            start_game,
            inputs=[player_prompt, neg_prompt, steps, cfg, width, height, mutation_rate,
                    freq_min, freq_max, save_mode, fix_seed, seed_input,
                    lora_selector, lora_weight, mutate_start, max_tags, tag_bias, n_choices,
                    show_tests, game_state],
            outputs=ALL_OUTPUTS,
        )

        def _stop_clicked(state):
            request_stop()
            return state, "⛔ Stop requested — finishing current image…"

        stop_btn.click(
            _stop_clicked,
            inputs=[game_state],
            outputs=[game_state, status_text],
        )

        def make_pick_fn(idx):
            def _pick(neg, steps, cfg, width, height, mut, fmin, fmax, save_mode, fix_seed, seed_val, max_tags, show_tests, state):
                for result in pick_winner(idx, neg, steps, cfg, width, height, mut,
                                          fmin, fmax, save_mode, fix_seed, seed_val, max_tags, show_tests, state):
                    # result = (*cards, *cols, *picks, *reacts, state, lineage, leaderboard, status)
                    yield (*result, gr.update(value="🔄 Restart"), gr.update(visible=True),
                           gr.update(visible=not fix_seed))
            return _pick

        PICK_INPUTS = [neg_prompt, steps, cfg, width, height, mutation_rate,
                       freq_min, freq_max, save_mode, fix_seed, seed_input, max_tags,
                       show_tests, game_state]

        for i, btn in enumerate(pick_btns):
            btn.click(
                make_pick_fn(i),
                inputs=PICK_INPUTS,
                outputs=[*PICK_OUTPUTS, start_btn, stop_btn, reroll_btn],
            )

        reroll_btn.click(
            reroll_round,
            inputs=[neg_prompt, steps, cfg, width, height, mutation_rate,
                    freq_min, freq_max, fix_seed, seed_input, max_tags,
                    show_tests, game_state],
            outputs=PICK_OUTPUTS,
        )

        def _gen_timeline(state):
            lineage = state.get("lineage", [])
            path = generate_timeline_image(lineage, state.get("starting_tokens", []),
                                           game_num=state.get("game", 1))
            if path is None:
                return gr.update(visible=False), "No rounds to draw yet — play some first."
            modes = {e.get("save_mode", "Don't save") for e in lineage}
            warn = ""
            if modes - {"Save all"}:
                warn = " ⚠️ Some rounds weren't saved with 'Save all' — unsaved images show as placeholders."
            try:
                os.startfile(path)  # Windows
            except AttributeError:
                import subprocess
                try:
                    subprocess.Popen(["xdg-open", path])
                except Exception:
                    subprocess.Popen(["open", path])
            except Exception:
                pass
            return gr.update(value=path, visible=True), f"🧬 Timeline saved: {path}{warn}"

        timeline_btn.click(_gen_timeline, inputs=[game_state],
                           outputs=[timeline_img, status_text])

        def _wipe_scores():
            tag_scores.clear_scores()
            return build_leaderboard_html(), "🗑️ Tag scores cleared."

        wipe_scores_btn.click(
            _wipe_scores,
            outputs=[leaderboard_display, status_text],
        )

        def _refresh_tests(show, state):
            """Re-render current cards and lineage when show_tests is toggled."""
            prompts = state.get("prompts", [])
            imgs    = state.get("imgs", [])
            width   = state.get("width", 512)
            height  = state.get("height", 512)
            cfg     = state.get("cfg", 7.0)
            seed    = state.get("seed", -1)
            neg_slot = state.get("neg_test_slot")
            pos_slot = state.get("pos_test_slot")
            if prompts:
                new_cards = cards_update(imgs, prompts, width, height, cfg, seed,
                                         show_tests=show, neg_slot=neg_slot, pos_slot=pos_slot)
            else:
                new_cards = [gr.update()] * 9
            lineage_html = build_lineage_html(state.get("lineage", []), show_tests=show)
            return (*new_cards, lineage_html)

        show_tests.change(
            _refresh_tests,
            inputs=[show_tests, game_state],
            outputs=[*card_htmls, lineage_display],
        )

        # Like/dislike wiring
        for i in range(9):
            like_btn    = react_btns[i * 2]
            dislike_btn = react_btns[i * 2 + 1]
            like_btn.click(
                lambda s, idx=i: react_image(idx, +1, s),
                inputs=[game_state],
                outputs=[game_state, like_btn, dislike_btn, status_text],
            )
            dislike_btn.click(
                lambda s, idx=i: react_image(idx, -1, s),
                inputs=[game_state],
                outputs=[game_state, like_btn, dislike_btn, status_text],
            )

    return [(ui, "The A/B Game", "the_ab_game_tab"),
            (build_scores_tab(), "📊 Tag Scores", "pe_tag_scores")]


script_callbacks.on_ui_tabs(on_ui_tabs)


# ---------------------------------------------------------------------------
# Force slider defaults into ui-config.json
# ---------------------------------------------------------------------------

UI_CONFIG_DEFAULTS = {
    "prompt_evolution/Width/value": 512,
    "prompt_evolution/Height/value": 512,
    "prompt_evolution/Steps/value": 20,
    "prompt_evolution/CFG scale/value": 7.0,
    "prompt_evolution/Mutations per image (avg)/value": 3,
    "prompt_evolution/Min post count (log)/value": 5,
    "prompt_evolution/Max post count (log)/value": 8,
    "prompt_evolution/Number of choices/value": 2,
}

def on_app_started(demo, app):
    try:
        import json
        ui_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "ui-config.json"
        )
        config = {}
        if os.path.exists(ui_config_path):
            with open(ui_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        changed = False
        for key, value in UI_CONFIG_DEFAULTS.items():
            if config.get(key) != value:
                config[key] = value
                changed = True
        if changed:
            with open(ui_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            print(f"[prompt-evolution] Updated ui-config.json")
    except Exception as e:
        print(f"[prompt-evolution] Could not update ui-config.json: {e}")

script_callbacks.on_app_started(on_app_started)