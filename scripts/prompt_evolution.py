"""
Prompt Evolution — A1111 Extension
====================================
A/B image game driven by genetic-style prompt mutation.
The winning prompt mutates each round using the active model's CLIP vocab,
filtered to meaningful tokens via wordfreq frequency scores.
"""

import os
import random
import datetime
import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks, shared


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

_vocab_cache: dict[str, list[str]] = {}


def find_danbooru_csv() -> str | None:
    """Look for danbooru.csv from the tagcomplete extension."""
    import pathlib
    our_dir = pathlib.Path(__file__).parent.parent
    extensions_root = our_dir.parent
    candidates = [
        extensions_root / "a1111-sd-webui-tagcomplete" / "tags" / "danbooru.csv",
        extensions_root / "sd-webui-tagcomplete" / "tags" / "danbooru.csv",
        extensions_root / "tagcomplete" / "tags" / "danbooru.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def build_clip_vocab_with_progress(freq_min: float, freq_max: float):
    """
    Generator: yields (status_str, progress 0-1, vocab_or_None).
    Uses danbooru.csv tag list by post count if tagcomplete is installed,
    falls back to wordfreq otherwise.
    """
    checkpoint_info = getattr(shared.sd_model, "sd_checkpoint_info", None)
    model_title = getattr(checkpoint_info, "title", None) or getattr(checkpoint_info, "name", None) or str(checkpoint_info)
    model_key = f"{model_title}_{freq_min}_{freq_max}"
    print(f"[prompt-evolution] Using model: {model_title}")

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
        print(f"[prompt-evolution] Could not read CLIP tokenizer: {e}")
        yield f"⚠️ {e}", 1.0, []
        return

    # Sliders map to log10(post_count): slider 3 = 1k posts, slider 6 = 1M posts
    import math
    lo = 10 ** min(freq_min, freq_max)
    hi = 10 ** max(freq_min, freq_max)

    # -- Try danbooru.csv -------------------------------------------------
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
                    if tag_type != 0:
                        continue  # general tags only
                    if not (lo <= post_count <= hi):
                        continue
                    name = row[0].strip()
                    name_plain = name.replace("_", " ")
                    if name.lower() in clip_vocab or name_plain.lower() in clip_vocab:
                        tags.append(name_plain)

            if tags:
                _vocab_cache[model_key] = tags
                print(f"[prompt-evolution] Danbooru vocab: {len(tags)} tags (post count {int(lo):,}–{int(hi):,})")
                yield f"✅ Vocab ready — {len(tags):,} danbooru tags", 1.0, tags
                return
            print("[prompt-evolution] Danbooru filter returned 0 tags, falling back to wordfreq")
        except Exception as e:
            print(f"[prompt-evolution] Danbooru CSV failed: {e}, falling back to wordfreq")

    # -- Fallback: wordfreq -----------------------------------------------
    yield "📖 Loading wordfreq table…", 0.1, None
    try:
        from wordfreq import get_frequency_dict
    except ImportError:
        yield "⚠️ No danbooru.csv and wordfreq not installed — using fallback vocab", 1.0, ["art", "portrait", "landscape", "fantasy", "vivid"]
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
    print(f"[prompt-evolution] wordfreq vocab: {len(clean)} tokens Zipf {freq_lo:.1f}–{freq_hi:.1f}")
    yield f"✅ Vocab ready — {len(clean):,} tokens (wordfreq)", 1.0, clean


def get_clip_vocab(freq_min: float, freq_max: float) -> list[str]:
    checkpoint_info = getattr(shared.sd_model, "sd_checkpoint_info", None)
    model_title = getattr(checkpoint_info, "title", None) or getattr(checkpoint_info, "name", None) or str(checkpoint_info)
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

def mutate_prompt(prompt: str, vocab: list[str], mutation_rate: int, locked: list[str] = None) -> str:
    tokens = [t.strip() for t in prompt.split(",") if t.strip()]
    locked_set = set(t.strip().lower() for t in locked) if locked else set()

    for _ in range(mutation_rate):
        if not vocab:
            break
        new_token = random.choice(vocab)
        mutable = [i for i, t in enumerate(tokens) if t.lower() not in locked_set]

        # Bias toward ADD when prompt is short, 50/50 at 8+ tokens
        token_count = len(tokens)
        if token_count < 8:
            add_chance = 1.0 - (token_count / 8) * 0.5  # 100% at 0, 50% at 8
        else:
            add_chance = 0.5

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


def save_winner(image, prompt: str, game_num: int, round_num: int) -> str | None:
    """Save the PIL image with game+round stamped filename. Returns save_dir or None."""
    try:
        save_dir = get_save_dir()
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"game{game_num:03d}_round{round_num:03d}_{timestamp}.png"
        path = os.path.join(save_dir, filename)
        image.save(path)
        print(f"[prompt-evolution] Saved winner: {path}")
        return save_dir
    except Exception as e:
        print(f"[prompt-evolution] Save failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_image(prompt: str, neg_prompt: str, steps: int, cfg: float, width: int, height: int, seed: int = -1):
    from modules.processing import StableDiffusionProcessingTxt2Img, process_images
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        prompt=prompt,
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
    p.script_args = [None] * scripts.scripts_txt2img.constructor_args_count if hasattr(scripts.scripts_txt2img, 'constructor_args_count') else []
    processed = process_images(p)
    return processed.images[0] if processed.images else None


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

def initial_state():
    return {
        "round": 0,
        "prompt_a": "",
        "prompt_b": "",
        "img_a": None,   # PIL image — stored so we can save the actual generated image
        "img_b": None,   # PIL image
        "winner_prompt": "",
        "lineage": [],
        "save_dir": None,
        "locked": [],
        "game": 1,
    }


# ---------------------------------------------------------------------------
# Round logic
# ---------------------------------------------------------------------------

def start_game(player_prompt, neg_prompt, steps, cfg, width, height, mutation_rate, freq_min, freq_max, auto_save, fix_seed, seed_val, state):
    """
    Outputs: img_a, img_b, prompt_a_text, prompt_b_text,
             pick_a, pick_b, status, game_state, start_btn
    """
    NO_IMAGES = (
        None, None, "", "",
        gr.update(visible=False), gr.update(visible=False),
    )
    BTN_WAIT = gr.update()  # no change during loading
    BTN_DONE = gr.update(value="🔄 Restart")

    vocab = None
    for status, progress, result in build_clip_vocab_with_progress(freq_min, freq_max):
        yield (*NO_IMAGES, status, state, BTN_WAIT)
        if result is not None:
            vocab = result

    if not vocab:
        yield (*NO_IMAGES, "⚠️ Could not build vocab — is a model loaded?", state, BTN_WAIT)
        return

    base = seed_prompt(player_prompt, vocab)
    locked = [t.strip() for t in player_prompt.split(",") if t.strip()]

    # Increment game number from previous state
    game_num = state.get("game", 1)
    if state.get("round", 0) > 0:
        game_num += 1
    split = random.randint(1, max(1, mutation_rate - 1)) if mutation_rate > 1 else 1
    prompt_a = mutate_prompt(base, vocab, split, locked)
    prompt_b = mutate_prompt(base, vocab, mutation_rate - split, locked)

    seed = int(seed_val) if fix_seed else -1

    yield (*NO_IMAGES, "🖼️ Generating image A…", state, BTN_WAIT)
    img_a = generate_image(prompt_a, neg_prompt, steps, cfg, width, height, seed)

    yield (img_a, None, prompt_a, "", gr.update(visible=False), gr.update(visible=False), "🖼️ Generating image B…", state, BTN_WAIT)
    img_b = generate_image(prompt_b, neg_prompt, steps, cfg, width, height, seed)

    state = initial_state()
    state["round"] = 1
    state["game"] = game_num
    state["prompt_a"] = prompt_a
    state["prompt_b"] = prompt_b
    state["img_a"] = img_a
    state["img_b"] = img_b
    state["locked"] = locked

    yield (
        img_a, img_b,
        prompt_a, prompt_b,
        gr.update(visible=True), gr.update(visible=True),
        "✅ Round 1 — pick the better image!",
        state,
        BTN_DONE,
    )


def pick_winner(choice, neg_prompt, steps, cfg, width, height, mutation_rate, freq_min, freq_max, auto_save, fix_seed, seed_val, state):
    vocab = get_clip_vocab(freq_min, freq_max)
    seed = int(seed_val) if fix_seed else -1

    winner_prompt = state["prompt_a"] if choice == "A" else state["prompt_b"]
    winner_img    = state["img_a"]    if choice == "A" else state["img_b"]
    round_num = state["round"]

    state["lineage"].append(winner_prompt)
    state["winner_prompt"] = winner_prompt
    state["round"] += 1

    # Save the actual generated PIL image — no re-generation needed
    save_msg = ""
    if auto_save and winner_img is not None:
        save_dir = save_winner(winner_img, winner_prompt, state.get("game", 1), round_num)
        if save_dir:
            state["save_dir"] = save_dir
            save_msg = " — winner saved"

    split = random.randint(1, max(1, mutation_rate - 1)) if mutation_rate > 1 else 1
    locked = state.get("locked", [])
    prompt_a = mutate_prompt(winner_prompt, vocab, split, locked)
    prompt_b = mutate_prompt(winner_prompt, vocab, mutation_rate - split, locked)

    yield (None, None, prompt_a, prompt_b, state, "", f"**Round {state['round']}** — generating A…")
    img_a = generate_image(prompt_a, neg_prompt, steps, cfg, width, height, seed)

    yield (img_a, None, prompt_a, prompt_b, state, "", f"**Round {state['round']}** — generating B…")
    img_b = generate_image(prompt_b, neg_prompt, steps, cfg, width, height, seed)

    lineage_text = "\n".join(f"Round {i+1}: {p}" for i, p in enumerate(state["lineage"]))
    state["prompt_a"] = prompt_a
    state["prompt_b"] = prompt_b
    state["img_a"] = img_a
    state["img_b"] = img_b

    yield (img_a, img_b, prompt_a, prompt_b, state, lineage_text, f"**Round {state['round']}** — pick the better image!{save_msg}")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def on_ui_tabs():
    css = """
    #pe-layout { display: flex !important; flex-direction: row !important; gap: 12px; align-items: flex-start; }
    #pe-left { display: flex; flex-direction: column; gap: 8px; width: 340px; min-width: 340px; height: 100vh; max-height: 100vh; overflow: hidden; }
    #pe-right { flex: 1; min-width: 0; max-width: 100%; overflow: hidden; overflow-y: auto; max-height: 100vh; box-sizing: border-box; }
    #pe-img-a, #pe-img-b { flex: 1; min-height: 0; overflow: hidden; }
    #pe-img-a img, #pe-img-b img { width: 100% !important; height: 100% !important; object-fit: contain !important; }
    #pe-img-a .image-container, #pe-img-b .image-container { height: 100% !important; }
    #pe-pick-a, #pe-pick-b { flex-shrink: 0; }
    #pe-status textarea { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #pe-right * { max-width: 100%; box-sizing: border-box; }
    #pe-right .gradio-slider { max-width: 100%; }
    """

    with gr.Blocks(analytics_enabled=False) as ui:
        gr.HTML(f"<style>{css}</style>")
        game_state = gr.State(initial_state())

        with gr.Row(elem_id="pe-layout"):

            # ── Left column: images ──────────────────────────────────────────
            with gr.Column(elem_id="pe-left"):
                img_a = gr.Image(label="Image A", interactive=False, type="pil", elem_id="pe-img-a")
                prompt_a_text = gr.Textbox(label="Prompt A", visible=False, interactive=False)
                pick_a = gr.Button("✅ Pick A", visible=False, elem_id="pe-pick-a")

                img_b = gr.Image(label="Image B", interactive=False, type="pil", elem_id="pe-img-b")
                prompt_b_text = gr.Textbox(label="Prompt B", visible=False, interactive=False)
                pick_b = gr.Button("✅ Pick B", visible=False, elem_id="pe-pick-b")

            # ── Right column: controls ───────────────────────────────────────
            with gr.Column(elem_id="pe-right"):
                gr.Markdown("## The A/B Game")

                start_btn = gr.Button("▶ Start Game", variant="primary")

                status_text = gr.Textbox(label="Status", interactive=False, elem_id="pe-status", max_lines=1)

                with gr.Accordion("⚙️ Settings", open=True):
                    with gr.Row():
                        show_prompts = gr.Checkbox(label="Show prompts", value=False)
                        auto_save = gr.Checkbox(label="💾 Auto-save winners", value=True)
                        open_folder_btn = gr.Button("📂 Open output folder", size="sm")

                    player_prompt = gr.Textbox(
                        label="Starting prompt (leave blank for random)",
                        placeholder="e.g. ancient temple, misty forest …",
                    )
                    neg_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="ugly, blurry, watermark …",
                    )
                    steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                    cfg = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFG scale")
                    with gr.Row():
                        width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                        height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                    mutation_rate = gr.Slider(1, 10, value=3, step=1, label="Mutations per round")
                    with gr.Row():
                        fix_seed = gr.Checkbox(label="Fix seed", value=False)
                        seed_input = gr.Number(label="Seed", value=42, precision=0, visible=False)
                    fix_seed.change(lambda x: gr.update(visible=x), inputs=[fix_seed], outputs=[seed_input])
                    with gr.Row():
                        freq_min = gr.Slider(1, 8, value=5, step=0.5, label="Min post count (log: 3=1k, 5=100k)")
                        freq_max = gr.Slider(1, 8, value=8, step=0.5, label="Max post count (log: 5=100k, 7=10M)")

        # ── Below layout: round label + lineage (always visible) ─────────────
        round_label = gr.Markdown("")
        with gr.Accordion("📜 Winning prompt lineage", open=False):
            lineage_box = gr.Textbox(label="", interactive=False, lines=8,
                placeholder="Winning prompts will appear here after each round …")

        # ── Wiring ──────────────────────────────────────────────────────────

        show_prompts.change(
            lambda show: (gr.update(visible=show), gr.update(visible=show)),
            inputs=[show_prompts],
            outputs=[prompt_a_text, prompt_b_text],
        )

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

        start_btn.click(
            start_game,
            inputs=[player_prompt, neg_prompt, steps, cfg, width, height, mutation_rate, freq_min, freq_max, auto_save, fix_seed, seed_input, game_state],
            outputs=[img_a, img_b, prompt_a_text, prompt_b_text, pick_a, pick_b, status_text, game_state, start_btn],
        )

        def pick_and_update(choice):
            def _pick(neg, steps, cfg, width, height, mut, freq_min, freq_max, auto_save, fix_seed, seed_val, state):
                for vals in pick_winner(choice, neg, steps, cfg, width, height, mut, freq_min, freq_max, auto_save, fix_seed, seed_val, state):
                    yield (*vals, gr.update(value="🔄 Restart"))
            return _pick

        pick_a.click(
            pick_and_update("A"),
            inputs=[neg_prompt, steps, cfg, width, height, mutation_rate, freq_min, freq_max, auto_save, fix_seed, seed_input, game_state],
            outputs=[img_a, img_b, prompt_a_text, prompt_b_text, game_state, lineage_box, round_label, start_btn],
        )

        pick_b.click(
            pick_and_update("B"),
            inputs=[neg_prompt, steps, cfg, width, height, mutation_rate, freq_min, freq_max, auto_save, fix_seed, seed_input, game_state],
            outputs=[img_a, img_b, prompt_a_text, prompt_b_text, game_state, lineage_box, round_label, start_btn],
        )

    return [(ui, "Prompt Evolution", "prompt_evolution_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)


# ---------------------------------------------------------------------------
# Force our slider defaults into ui-config.json so they can't be overridden
# ---------------------------------------------------------------------------

UI_CONFIG_DEFAULTS = {
    "prompt_evolution/Width/value": 768,
    "prompt_evolution/Height/value": 768,
    "prompt_evolution/Steps/value": 20,
    "prompt_evolution/CFG scale/value": 7.0,
    "prompt_evolution/Mutations per round/value": 3,
    "prompt_evolution/Min post count (log: 3=1k, 5=100k)/value": 5,
    "prompt_evolution/Max post count (log: 5=100k, 7=10M)/value": 8,
}

def on_app_started(demo, app):
    try:
        import json
        ui_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "ui-config.json"
        )
        if os.path.exists(ui_config_path):
            with open(ui_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        changed = False
        for key, value in UI_CONFIG_DEFAULTS.items():
            if config.get(key) != value:
                config[key] = value
                changed = True

        if changed:
            with open(ui_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            print(f"[prompt-evolution] Updated ui-config.json with default values")

    except Exception as e:
        print(f"[prompt-evolution] Could not update ui-config.json: {e}")

script_callbacks.on_app_started(on_app_started)
