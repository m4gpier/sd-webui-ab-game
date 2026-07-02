# sd-webui-ab-game — Prompt Evolution

An [AUTOMATIC1111 / reForge] extension that evolves Stable Diffusion prompts through gameplay. Each round generates 2–9 mutated variants of the current winner — you pick the best image, and the prompt evolves. Along the way, the extension learns which tags you like and dislike, and uses that knowledge to steer future mutations.

## Installation

Clone into your `extensions` folder:

```
git clone https://github.com/m4gpier/sd-webui-ab-game extensions/sd-webui-ab-game
```

Restart the WebUI. Two new tabs appear: **The A/B Game** and **📊 Tag Scores**.

Optional: install a tagcomplete extension (its `tags/danbooru.csv` is auto-detected) for a full danbooru mutation vocabulary, including multi-word tags like `long hair` and `looking at viewer`.

## Playing

1. Enter a starting prompt (or leave blank for random) and hit **▶ Start Game**.
2. Each round shows N images. **Pick** the best; optionally **👍 / 👎** individual images.
3. The winner becomes the parent of the next generation of mutations.
4. **🎲 Reroll round** regenerates the current round without scoring (disabled when the seed is fixed).
5. **🧬 Generate Timeline** renders the whole game as one PNG family tree — full-size images, seeds, prompts with changed tags highlighted, like/dislike and test badges, and winner lines. Opens automatically and saves into the game folder.

### Options

- **Number of choices** (2–9), mutation rate, max tags, resolution/steps/CFG
- **Mutate starting tokens** — your starting prompt is always the round-1 base; this checkbox controls whether those tokens may later be mutated
- **Tag bias** (0–100%) — how strongly mutations favour tags you've scored well
- **LoRAs** — applied to every generation, never mutated
- **Fixed seed** for apples-to-apples comparisons
- **Save mode** — *Save all* / *Save winner* / *Don't save*

### Output folders

```
outputs/txt2img-images/prompt-evolution/
  YYYYMMDD/              ← day folder
    G001/                ← game folder
      HHMMSS_R001_A.png  ← every image, time-marker titles
      timeline_HHMMSS.png
```

Note: the timeline reads images from disk, so use **Save all** for a complete tree — unsaved slots render as "Image Not Available".

## Tag scoring

The extension keeps a persistent score per tag (`tag_scores.json`, next to the output folders — legacy `.txt` files migrate automatically):

- **Win credit** — tags the winner *added* vs. its parent split +1 between them (1 new tag → +1; k new tags → +1/k each)
- **Likes / dislikes** — same dilution, applied to the tags new in that image
- **Mild penalties** — tags the winner dropped ("swap-out") and mutations in losing images ("not-picked") take small, configurable hits
- **Decay** — optionally fade all scores each round so stale preferences don't linger
- **Touches** — each tag tracks how often it's been judged; rarely-seen tags are damped in the mutation pool (confidence weighting) and shown as ×N in the chart
- **Suppression** — negatively-scored tags are increasingly unlikely to appear in random mutations (a −10 tag essentially never shows up outside a test)
- **Exploration bonus** — a slice of mutations prefers never-scored tags so the system keeps discovering

### Negativity / positivity tests

Periodically (every N generations, configurable), one slot becomes a hidden test: the previous winner's prompt plus exactly one scored tag — nothing else changed. Selection favours extreme, under-tested tags. Your reaction confirms (±test strength) or softens (halves) the tag's score. Enable **Show test markers** to see which slot is a test.

## 📊 Tag Scores tab

- **Bar chart** of top/bottom tags (score + ×touches)
- **Trend lines** — score history per round; auto-plots the top/bottom N (slider 5–50) or hand-picked tags
- **Manual editing** — ±1 / set exact / delete a tag, or edit the capped, searchable table directly
- **🧹 Prune** zero-score tags, **🗑️ Wipe** everything
- **⚙️ Pool tuning** — decay, confidence k, swap-out & not-picked penalties, test strength & interval, like/dislike weights, exploration bonus. All persist in the JSON.

## Files

- `scripts/prompt_evolution.py` — game, UI, mutation engine, timeline renderer
- `tag_scores.py` — persistent scoring backend (JSON, thread-safe)
