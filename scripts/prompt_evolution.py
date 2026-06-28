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
                    name_plain = name.replace("_", " ")
                    if name.lower() in clip_vocab or name_plain.lower() in clip_vocab:
                        tags.append(name_plain)
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

def _pick_token(vocab: list[str], bias: float,
                pool_tags: list[str], pool_weights: list[float]) -> str:
    """
    Choose a new token. With probability `bias`, draw from the score-weighted
    tracked-tag pool; otherwise draw uniformly from the full vocab.
    Falls back to uniform if the weighted pool is empty.
    """
    if bias > 0 and pool_tags and random.random() < bias:
        return random.choices(pool_tags, weights=pool_weights, k=1)[0]
    return random.choice(vocab)


def mutate_prompt(prompt: str, vocab: list[str], mutation_rate: int,
                  locked: list[str] = None, max_tags: int = 75,
                  bias: float = 0.0,
                  pool_tags: list[str] = None, pool_weights: list[float] = None) -> str:
    tokens = [t.strip() for t in prompt.split(",") if t.strip()]
    locked_set = set(t.strip().lower() for t in locked) if locked else set()
    pool_tags = pool_tags or []
    pool_weights = pool_weights or []

    for _ in range(mutation_rate):
        if not vocab:
            break
        new_token = _pick_token(vocab, bias, pool_tags, pool_weights)
        mutable = [i for i, t in enumerate(tokens) if t.lower() not in locked_set]

        token_count = len(tokens)
        if token_count >= max_tags:
            add_chance = 0.0
        else:
            add_chance = 1.0 - (token_count / max_tags)

        if not mutable or random.random() < add_chance:
            tokens.append(new_token)
        else:
            idx = random.choice(mutable)
            tokens[idx] = new_token

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


def save_images(imgs: list, labels: list[str], game_num: int, round_num: int) -> list[str]:
    """Save a list of PIL images, returns list of saved paths."""
    paths = []
    try:
        save_dir = get_save_dir()
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for img, label in zip(imgs, labels):
            path = os.path.join(save_dir, f"game{game_num:03d}_round{round_num:03d}_{label}_{timestamp}.png")
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
                           cfg: float, seed: int, generating: bool = False) -> str:
    """Single image card HTML — no buttons (those are Gradio components beneath)."""
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
    meta = (
        f"<div style='font-size:9px;color:#666;margin-top:3px;text-align:center;'>"
        f"{width}×{height} · CFG {cfg} · {seed_str}</div>"
    )
    prompt_div = (
        f"<div style='font-size:9px;color:#555;margin-top:3px;line-height:1.3;"
        f"overflow-wrap:anywhere;'>{prompt}</div>"
        if prompt else ""
    )
    return img_tag + meta + prompt_div


# ---------------------------------------------------------------------------
# Lineage HTML — one row per round, N thumbnails, winner blue-highlighted
# ---------------------------------------------------------------------------

def build_lineage_html(lineage: list[dict]) -> str:
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
            label = LABELS[i] if i < len(LABELS) else str(i + 1)
            border = "2px solid #4a9eff" if is_winner else "1px solid #333"
            lbl_col = "#4a9eff" if is_winner else "#666"
            img_tag = (
                f"<img src='{uri}' style='width:{THUMB}px;height:auto;border-radius:4px;display:block;margin:0 auto;' />"
                if uri else
                f"<div style='width:{THUMB}px;height:{THUMB}px;background:#1a1a1a;border-radius:4px;margin:0 auto;'></div>"
            )
            winner_badge = "<div style='font-size:8px;color:#4a9eff;text-align:center;margin-top:2px;font-weight:600;'>WINNER</div>" if is_winner else ""
            cards.append(f"""
            <div style="flex:1 1 0;min-width:0;display:flex;flex-direction:column;align-items:center;
                        gap:3px;padding:5px;border:{border};border-radius:8px;
                        background:#111;box-sizing:border-box;overflow:hidden;">
              <div style="font-size:10px;font-weight:600;color:{lbl_col};">{label}</div>
              {img_tag}
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
            bsign = "+" if bscore > 0 else ""
            pill_bg, badge_bg, txt = colors(bscore)
            left = f"""<div style="display:flex;align-items:center;gap:0;flex:1;min-width:0;justify-content:flex-end;">
              <div style="padding:5px 10px;display:flex;align-items:center;min-width:0;overflow:hidden;">
                <span style="font-size:13px;font-weight:500;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{btag}</span>
              </div>
              <div style="padding:5px 7px;background:{pill_bg};border-radius:6px 0 0 6px;display:flex;align-items:center;gap:4px;flex-shrink:0;">
                <span style="font-size:12px;font-weight:700;color:{txt};">{bsign}{bscore}</span>
                <div style="width:20px;height:20px;background:{badge_bg};border-radius:3px;display:flex;align-items:center;justify-content:center;">
                  <span style="font-size:10px;font-weight:700;color:{txt};">{rank}</span>
                </div>
              </div>
            </div>"""
        else:
            left = "<div style='flex:1;'></div>"

        if i < len(top):
            ttag, tscore = top[i]
            tsign = "+" if tscore > 0 else ""
            pill_bg, badge_bg, txt = colors(tscore)
            right = f"""<div style="display:flex;align-items:center;gap:0;flex:1;min-width:0;">
              <div style="padding:5px 7px;background:{pill_bg};border-radius:0 6px 6px 0;display:flex;align-items:center;gap:4px;flex-shrink:0;">
                <div style="width:20px;height:20px;background:{badge_bg};border-radius:3px;display:flex;align-items:center;justify-content:center;">
                  <span style="font-size:10px;font-weight:700;color:{txt};">{rank}</span>
                </div>
                <span style="font-size:12px;font-weight:700;color:{txt};">{tsign}{tscore}</span>
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
    }


# ---------------------------------------------------------------------------
# Mutation helpers for N choices
# ---------------------------------------------------------------------------

def generate_n_prompts(base: str, vocab: list[str], n: int,
                       mutation_rate: int, locked: list[str], max_tags: int,
                       bias: float = 0.0) -> list[str]:
    """
    Generate N prompts by mutating base. Each gets a random number of
    mutations in [1, mutation_rate*2-1] averaging to mutation_rate.
    If bias > 0, a fraction of new tokens are drawn from the score-weighted tag pool.
    """
    pool_tags, pool_weights = ([], [])
    if bias > 0:
        pool_tags, pool_weights = tag_scores.weighted_pool()

    prompts = []
    for _ in range(n):
        m = random.randint(1, max(1, mutation_rate * 2 - 1))
        prompts.append(mutate_prompt(base, vocab, m, locked, max_tags,
                                     bias=bias, pool_tags=pool_tags, pool_weights=pool_weights))
    return prompts


# ---------------------------------------------------------------------------
# Pick button HTML builder — N dynamic buttons
# ---------------------------------------------------------------------------

def cards_update(imgs: list, prompts: list, width: int, height: int,
                  cfg: float, seed: int, generating_idx: int = -1) -> list:
    """Return 9 gr.update(value=...) for card HTML slots."""
    updates = []
    for i in range(9):
        if i < len(prompts):
            html = build_image_card_html(
                imgs[i] if i < len(imgs) else None,
                prompts[i],
                width, height, cfg, seed,
                generating=(i == generating_idx),
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
               tag_bias, n_choices, state):
    n = int(n_choices)
    EMPTY_CARDS = cards_update([], [], width, height, cfg, -1)
    NO_COL  = col_visibility(n)
    NO_PICK = pick_buttons_update(n, False)
    NO_REACT = react_buttons_update(n, False)

    vocab = None
    for status, progress, result in build_clip_vocab_with_progress(freq_min, freq_max):
        yield (*EMPTY_CARDS, *NO_COL, *NO_PICK, *NO_REACT, status, state, gr.update(), build_lineage_html([]))
        if result is not None:
            vocab = result

    if not vocab:
        yield (*EMPTY_CARDS, *NO_COL, *NO_PICK, *NO_REACT,
               "⚠️ Could not build vocab — is a model loaded?",
               state, gr.update(), build_lineage_html([]))
        return

    base = seed_prompt(player_prompt, vocab)
    starting_tokens = [t.strip() for t in player_prompt.split(",") if t.strip()]
    locked = [] if mutate_start else starting_tokens

    game_num = state.get("game", 1)
    if state.get("round", 0) > 0:
        game_num += 1

    seed = int(seed_val) if fix_seed else -1
    lora_list = lora_selections if isinstance(lora_selections, list) else ([lora_selections] if lora_selections else [])

    bias = tag_bias / 100.0
    prompts = generate_n_prompts(base, vocab, n, mutation_rate, locked, max_tags, bias=bias)
    imgs = [None] * n

    for i in range(n):
        cur_cards = cards_update(imgs, prompts, width, height, cfg, seed, generating_idx=i)
        yield (*cur_cards, *NO_COL, *NO_PICK, *NO_REACT,
               f"Round 1 · generating {LABELS[i]}…", state, gr.update(), build_lineage_html([]))
        imgs[i] = generate_image(prompts[i], neg_prompt, steps, cfg, width, height,
                                  seed, lora_list, lora_weight)

    state = initial_state()
    state["round"] = 1
    state["game"] = game_num
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

    ready_cards = cards_update(imgs, prompts, width, height, cfg, seed)
    yield (*ready_cards,
           *col_visibility(n),
           *pick_buttons_update(n, True),
           *react_buttons_update(n, True),
           "Round 1 · pick the best image",
           state,
           gr.update(value="🔄 Restart"),
           build_lineage_html([]))


# ---------------------------------------------------------------------------
# Round logic — pick_winner
# ---------------------------------------------------------------------------

def pick_winner(winner_idx: int, neg_prompt, steps, cfg, width, height, mutation_rate,
                freq_min, freq_max, save_mode, fix_seed, seed_val, max_tags, state):
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

    # Commit buffered like/dislike reactions
    for i, reaction in enumerate(reactions):
        if reaction is not None and i < len(prompts):
            tag_scores.adjust_tags(prompts[i], reaction, extra_tags=starting_set)

    # Thumbnails for lineage
    thumb_uris = [pil_to_thumbnail_uri(img) for img in imgs]

    # Save
    save_msg = ""
    if save_mode == "Save all":
        paths = save_images(imgs, [LABELS[i] for i in range(len(imgs))], game_num, round_num)
        if paths:
            state["save_dir"] = os.path.dirname(paths[0])
            save_msg = f" · {len(paths)} saved"
    elif save_mode == "Save winner":
        paths = save_images([imgs[winner_idx]], [f"winner{LABELS[winner_idx]}"], game_num, round_num)
        if paths:
            state["save_dir"] = os.path.dirname(paths[0])
            save_msg = f" · winner ({LABELS[winner_idx]}) saved"

    lineage_entry = {
        "round": round_num,
        "winner_idx": winner_idx,
        "prompts": list(prompts),
        "thumb_uris": thumb_uris,
        "loras": list(lora_list),
        "lora_weight": lora_weight,
    }
    state["lineage"].append(lineage_entry)
    state["winner_prompt"] = winner_prompt
    state["round"] += 1
    state["width"] = width
    state["height"] = height
    state["cfg"] = cfg
    state["seed"] = seed

    # Record tag scores: winner vs each loser
    for i, p in enumerate(prompts):
        if i != winner_idx:
            tag_scores.record_round(winner_prompt, p)

    next_round = state["round"]
    locked = [] if state.get("mutate_start") else state.get("locked", [])
    bias = state.get("tag_bias", 0) / 100.0
    new_prompts = generate_n_prompts(winner_prompt, vocab, n, mutation_rate, locked, max_tags, bias=bias)
    new_imgs = [None] * n

    lineage_html     = build_lineage_html(state["lineage"])
    leaderboard_html = build_leaderboard_html()
    NO_COL   = col_visibility(n)
    NO_PICK  = pick_buttons_update(n, False)
    NO_REACT = react_buttons_update(n, False)

    for i in range(n):
        cur_cards = cards_update(new_imgs, new_prompts, width, height, cfg, seed, generating_idx=i)
        yield (*cur_cards, *NO_COL, *NO_PICK, *NO_REACT,
               state, lineage_html, leaderboard_html,
               f"Round {next_round} · generating {LABELS[i]}…")
        new_imgs[i] = generate_image(new_prompts[i], neg_prompt, steps, cfg, width, height,
                                      seed, lora_list, lora_weight)

    state["prompts"]   = new_prompts
    state["imgs"]      = new_imgs
    state["reactions"] = [None] * n

    ready_cards = cards_update(new_imgs, new_prompts, width, height, cfg, seed)
    yield (*ready_cards,
           *col_visibility(n),
           *pick_buttons_update(n, True),
           *react_buttons_update(n, True),
           state, lineage_html, leaderboard_html,
           f"Round {next_round} · pick the best image{save_msg}")


# ---------------------------------------------------------------------------
# React (like/dislike) handler
# ---------------------------------------------------------------------------

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
                gr.HTML("<div style='font-size:13px;font-weight:500;color:var(--text-secondary);margin-bottom:6px;'>📜 Winning prompt lineage</div>")
                lineage_display = gr.HTML(value=build_lineage_html([]), elem_id="pe-lineage")
            with gr.Column(elem_id="pe-bottom-right"):
                gr.HTML("<div style='font-size:13px;font-weight:500;color:var(--text-secondary);margin-bottom:6px;'>🏆 Tag leaderboard</div>")
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
                        status_text, game_state, start_btn, lineage_display]
        PICK_OUTPUTS = [*card_htmls, *card_cols, *pick_btns, *react_btns,
                        game_state, lineage_display, leaderboard_display, status_text]

        start_btn.click(
            start_game,
            inputs=[player_prompt, neg_prompt, steps, cfg, width, height, mutation_rate,
                    freq_min, freq_max, save_mode, fix_seed, seed_input,
                    lora_selector, lora_weight, mutate_start, max_tags, tag_bias, n_choices,
                    game_state],
            outputs=ALL_OUTPUTS,
        )

        def make_pick_fn(idx):
            def _pick(neg, steps, cfg, width, height, mut, fmin, fmax, save_mode, fix_seed, seed_val, max_tags, state):
                for result in pick_winner(idx, neg, steps, cfg, width, height, mut,
                                          fmin, fmax, save_mode, fix_seed, seed_val, max_tags, state):
                    # result = (*cards, *cols, *picks, *reacts, state, lineage, leaderboard, status)
                    # PICK_OUTPUTS doesn't include start_btn, so append it
                    yield (*result, gr.update(value="🔄 Restart"))
            return _pick

        PICK_INPUTS = [neg_prompt, steps, cfg, width, height, mutation_rate,
                       freq_min, freq_max, save_mode, fix_seed, seed_input, max_tags, game_state]

        for i, btn in enumerate(pick_btns):
            btn.click(
                make_pick_fn(i),
                inputs=PICK_INPUTS,
                outputs=[*PICK_OUTPUTS, start_btn],
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

    return [(ui, "The A/B Game", "the_ab_game_tab")]


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