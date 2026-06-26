# The A/B Game

A prompt evolution game for Stable Diffusion WebUI. Pick the better image each round, the winner mutates and spawns the next pair. Repeat until you get bored.

---

## How it works

Each round generates two images from slightly different prompts. You pick A or B. The winning prompt mutates, tokens are randomly added or swapped using tags from the model's actual vocabulary and the next round begins. Over time, the prompt evolves in the direction of your taste.

- **Both prompts mutate** from the winner each round, independently
- **Locked tokens** — anything in your starting prompt will remain
- **Short prompts grow first** — the mutation system biases toward adding new words until the prompt reaches ~8 tokens, then switches to a mix of adds and swaps
- **Mutations are split** between A and B, not doubled — setting 4 mutations might give A 3 and B 1, or A 1 and B 3

---

## Installation

**Option 1 — Via WebUI (recommended)**
1. Go to **Extensions → Install from URL**
2. Paste this repo's URL and click **Install**
3. Restart the WebUI

**Option 2 — Manual**
```bash
cd stable-diffusion-webui/extensions
git clone https://github.com/m4gpier/sd-webui-ab-game.git
```

Then restart the WebUI. `install.py` will automatically install `wordfreq` into the WebUI's venv on first launch.

**Recommended:** also install [sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git). When present, The A/B Game uses its Danbooru tag list as the mutation vocabulary since I designed this to work with Anime Models. Without it, the extension falls back to a general English frequency list.

---

## Compatibility

Tested on:
- **reForge** (primary)
- **AUTOMATIC1111** (should work)
- **SDXL models** (recommended — defaults tuned for SDXL)
- **SD 1.5 models** (supported, change width/height defaults accordingly)

---

## Settings

| Setting | Description |
|---|---|
| **Starting prompt** | Seeds the game. Leave blank for a random tag. Any words entered here are locked and cannot be mutated away. |
| **Negative prompt** | Applied to every generation. |
| **Steps / CFG** | Standard generation quality settings. |
| **Width / Height** | Image dimensions. Defaults to 768×768. |
| **Mutations per round** | Total mutations split between A and B each round. |
| **Min / Max post count** | Filters the Danbooru tag pool by popularity (log scale: 3 = 1k posts, 5 = 100k, 7 = 10M). Default range 5–8 keeps well-known tags only. |
| **Fix seed** | Lock the noise seed so differences between A and B come purely from the prompt, not randomness. |
| **Auto-save winners** | Automatically saves the picked image to `outputs/txt2img-images/prompt-evolution/`. |
| **Show prompts** | Reveal the prompt text under each image. Off by default for a cleaner experience. |

---

## Output

Winners are saved to:
```
outputs/txt2img-images/prompt-evolution/game001_round003_20240626-143022.png
```

Game number increments each time you hit Restart, so sessions are easy to browse.

---

## Vocabulary

The mutation word pool is built from:

1. **Danbooru tags** (if tagcomplete is installed) — filtered to general tags only (no character/artist/copyright names), matched against the active model's CLIP tokenizer, and filtered by post count range. This gives you words the model genuinely understands.
2. **wordfreq fallback** — if tagcomplete isn't installed, falls back to a general English frequency list filtered to the model's CLIP vocab.

The vocab is cached per model + settings combination.

---

## License

MIT
