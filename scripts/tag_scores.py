"""
Tag Scores — Prompt Evolution helper
=====================================
Tracks cumulative win/loss scores for prompt tags across games.

Scoring rule per round:
  - Each tag in the WINNER prompt: +1
  - Each tag in the LOSER prompt only: -1
  - Tags in BOTH prompts: net 0 (cancel out)

Scores persist to a plain TXT file (one "tag<TAB>score" line per tag),
loaded on first use and flushed after every update.
"""

import os
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# File location — sits next to the A1111 outputs/prompt-evolution folder
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_scores: dict[str, int] = {}
_scores_path: Optional[str] = None
_loaded = False


def _get_scores_path() -> str:
    global _scores_path
    if _scores_path is not None:
        return _scores_path
    try:
        from modules import shared
        base = shared.opts.outdir_txt2img_samples
    except Exception:
        base = os.path.join(os.path.expanduser("~"), "outputs", "txt2img-images")
    folder = os.path.join(base, "prompt-evolution")
    os.makedirs(folder, exist_ok=True)
    _scores_path = os.path.join(folder, "tag_scores.txt")
    return _scores_path


def _load() -> None:
    global _loaded
    if _loaded:
        return
    path = _get_scores_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                tag, _, raw = line.partition("\t")
                try:
                    _scores[tag.strip()] = int(raw.strip())
                except ValueError:
                    pass
    _loaded = True


def _flush() -> None:
    path = _get_scores_path()
    with open(path, "w", encoding="utf-8") as f:
        for tag, score in sorted(_scores.items()):
            f.write(f"{tag}\t{score}\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_round(winner_prompt: str, loser_prompt: str) -> None:
    """
    Update scores given the winner and loser prompts for one round.
      - Tags exclusive to winner: +1
      - Tags exclusive to loser:  -1
      - Tags in both:             net 0
    """
    def parse(prompt: str) -> set[str]:
        return {t.strip().lower() for t in prompt.split(",") if t.strip()}

    winner_tags = parse(winner_prompt)
    loser_tags  = parse(loser_prompt)

    with _lock:
        _load()
        for tag in winner_tags - loser_tags:
            _scores[tag] = _scores.get(tag, 0) + 1
        for tag in loser_tags - winner_tags:
            _scores[tag] = _scores.get(tag, 0) - 1
        _flush()


def adjust_tags(prompt: str, delta: int, extra_tags: set[str] | None = None) -> None:
    """
    Apply `delta` to ALL tags in `prompt`, plus any `extra_tags` (e.g. locked
    starting tokens that aren't in the evolved prompt string itself).
    Used by the like (+1) / dislike (-1) buttons.
    """
    def parse(p: str) -> set[str]:
        return {t.strip().lower() for t in p.split(",") if t.strip()}

    tags = parse(prompt)
    if extra_tags:
        tags |= {t.strip().lower() for t in extra_tags if t.strip()}

    with _lock:
        _load()
        for tag in tags:
            _scores[tag] = _scores.get(tag, 0) + delta
        _flush()


def get_scores() -> dict[str, int]:
    """Return a copy of the current scores dict."""
    with _lock:
        _load()
        return dict(_scores)


def top_n(n: int = 10) -> list[tuple[str, int]]:
    """Return the n highest-scoring tags, descending."""
    scores = get_scores()
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def bottom_n(n: int = 10) -> list[tuple[str, int]]:
    """Return the n lowest-scoring tags, ascending."""
    scores = get_scores()
    return sorted(scores.items(), key=lambda x: x[1])[:n]


def scores_path() -> str:
    return _get_scores_path()


def weighted_pool(top_boost: int = 10, boost_mult: float = 3.0) -> tuple[list[str], list[float]]:
    """
    Build a (tags, weights) pool for biased mutation, drawn from positively-scored
    tags only. Weight = score, with the top `top_boost` tags multiplied by
    `boost_mult` for extra emphasis. Returns ([], []) if no positive tags exist.
    """
    scores = get_scores()
    positive = {t: s for t, s in scores.items() if s > 0}
    if not positive:
        return [], []

    # Identify the top-N tags for the boost
    top_tags = set(t for t, _ in sorted(positive.items(), key=lambda x: x[1], reverse=True)[:top_boost])

    tags, weights = [], []
    for tag, score in positive.items():
        w = float(score)
        if tag in top_tags:
            w *= boost_mult
        tags.append(tag)
        weights.append(w)
    return tags, weights