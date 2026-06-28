# The A/B Game

A prompt evolution game for Stable Diffusion WebUI. Generate a set of images, pick the best one, and watch the prompt mutate and improve over rounds. Repeat until you're happy with the result.

---

## How it works

Each round generates N images from slightly mutated prompts. You pick the best one. That prompt becomes the parent for the next round, spawning N new children with fresh mutations. Over time, the prompt drifts toward whatever you keep picking.

- **2–9 choices per round** — configurable via the Number of choices slider
- **Each child mutates independently** — random number of mutations per image, averaging to your Mutations setting
- **Locked tokens** — your starting prompt words are pinned and can't be mutated away (unless you check *Mutate starting prompt*)
- **Soft tag cap** — as the prompt fills up, mutations increasingly swap existing tags instead of adding new ones. At the max, it's swaps only
- **Tag bias** — once tags have accumulated scores, you can bias mutations toward tokens that have historically won

---

## Installation

**Option 1 — Via WebUI (recommended)**

1. Go to **Extensions → Install from URL**
2. Paste this repo URL and click **Install**
3. Restart

**Option 2 — Manual**

```bash
cd stable-diffusion-webui/extensions
git clone https://github.com/m4gpier/sd-webui-ab-game.git
```

Restart the WebUI. `install.py` will handle installing `wordfreq` into the venv on first launch.

**Recommended:** install [sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git) alongside this. When present, the game uses its Danbooru tag list as the mutation vocabulary — tags the model actually understands, filtered by post count. Without it, it falls back to a general English word frequency list.

---

## Compatibility

Tested on:
- **reForge** (primary development target)
- **AUTOMATIC1111** (should work)
- **SDXL models** (recommended — defaults tuned for SDXL)
- **SD 1.5** (supported, adjust width/height accordingly)

---

## Settings

| Setting | Description |
|---|---|
| **Number of choices** | How many images to generate and compare each round (2–9). |
| **Starting prompt** | Seeds the first round. Leave blank for a random tag. Locked by default — won't mutate away. |
| **Mutate starting prompt** | If checked, starting tokens are no longer protected and can be swapped out. |
| **Negative prompt** | Applied to every generation. |
| **Steps / CFG** | Standard generation settings. |
| **Width / Height** | Image dimensions. Default 512×512. |
| **Mutations per image (avg)** | Average number of token mutations applied to each child prompt. Actual count varies randomly per image. |
| **Max tags** | Soft cap on prompt length. The closer you get, the more likely a mutation swaps instead of adds. At the cap, adds stop entirely. Default 20. |
| **Tag bias %** | At 0%, mutations are random. At higher values, a proportion of new tokens are drawn from the score-weighted leaderboard — favouring tags that have historically won. Default 0%. |
| **Fix seed** | Locks the noise seed so A/B differences come purely from the prompt. |
| **Min / Max post count** | Filters the Danbooru vocabulary by tag popularity (log scale: 3 = 1k posts, 5 = 100k, 7 = 10M). Default 5–8 keeps well-known tags. |
| **Save images** | *Save all* saves every image each round. *Save winner* saves only the picked image. |
| **LoRA** | Select one or more LoRAs to apply at a fixed weight every generation. LoRA tags are injected at generation time and not part of the mutation pool. |

---

## Tag scoring & leaderboard

Every round, the game tracks which tags won and which lost:

- Tags **exclusive to the winner** → +1
- Tags **exclusive to losers** → -1
- Tags in **both** → no change

The 👍/👎 buttons let you rate images independently of picking — all tags in that image get +1 or -1 applied when you pick. These scores persist across sessions in `outputs/txt2img-images/prompt-evolution/tag_scores.txt`.

The leaderboard shows the top and bottom 10 tags by cumulative score. The **tag bias** feature draws from this — positive-scored tags are weighted by score, with the top 10 getting a 3× boost.

---

## Output

Images are saved to:

```
outputs/txt2img-images/prompt-evolution/game001_round003_A_20240626-143022.png
```

Game number increments on each Restart. The **📜 Winning prompt lineage** panel below the grid shows every round's images with the winner highlighted in blue.

---

## Vocabulary

Mutation tokens are sourced from:

1. **Danbooru tags** — if tagcomplete is installed, general-category tags are filtered by post count range and matched against the active model's CLIP tokenizer. This keeps mutations to words the model genuinely knows.
2. **wordfreq fallback** — if tagcomplete isn't present, falls back to English words from the model's CLIP vocab filtered by Zipf frequency.

The vocab is cached per model + frequency settings, so it's only built once per session.

---

## License

MIT
