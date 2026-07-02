"""
Tag Scores — Prompt Evolution helper
=====================================
Tracks cumulative win/loss scores for prompt tags across games.

Scoring rule per round:
  - Each tag in the WINNER prompt: +1
  - Each tag in the LOSER prompt only: -1
  - Tags in BOTH prompts: net 0 (cancel out)

Persistence (v2): a JSON file holding {scores, touches, meta}.
  - scores:  {tag: float}      cumulative score
  - touches: {tag: int}        how many rounds/adjustments touched the tag
                               (a confidence signal — +5 over 50 touches is
                               more trustworthy than +5 over 1)
  - meta:    {rounds, ...}
Old tab-separated `tag_scores.txt` files are auto-migrated on first load.
"""

import os
import json
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Colour + part vocabulary (mirrors prompt_evolution.py)
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


def _split_color_part(tag: str) -> tuple[str | None, str | None]:
    """If tag is a colour+part compound (e.g. 'red eyes'), return (colour, part)."""
    parts = tag.strip().lower().split()
    if len(parts) == 2 and parts[0] in _COLORS and parts[1] in _PARTS:
        return parts[0], parts[1]
    return None, None


def _expand_tags(tags: set[str], weight: float) -> dict[str, float]:
    """Build a {tag: delta} map, scoring colour+part compounds as their components."""
    result: dict[str, float] = {}
    for tag in tags:
        colour, part = _split_color_part(tag)
        if colour:
            result[colour] = result.get(colour, 0) + weight
            result[part] = result.get(part, 0) + weight
        else:
            result[tag] = result.get(tag, 0) + weight
    return result

# ---------------------------------------------------------------------------
# State + file location
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_scores: dict[str, float] = {}
_touches: dict[str, int] = {}
_history: dict[str, list] = {}          # tag -> [[round, score], ...] (sparse)
_meta: dict = {"rounds": 0, "decay": 1.0}
_folder: Optional[str] = None
_loaded = False

_HISTORY_MAX_POINTS = 300               # per-tag ring buffer cap


def _get_folder() -> str:
    global _folder
    if _folder is not None:
        return _folder
    try:
        from modules import shared
        base = shared.opts.outdir_txt2img_samples
    except Exception:
        base = os.path.join(os.path.expanduser("~"), "outputs", "txt2img-images")
    _folder = os.path.join(base, "prompt-evolution")
    os.makedirs(_folder, exist_ok=True)
    return _folder


def _get_scores_path() -> str:
    return os.path.join(_get_folder(), "tag_scores.json")


def _legacy_txt_path() -> str:
    return os.path.join(_get_folder(), "tag_scores.txt")


def _load() -> None:
    global _loaded
    if _loaded:
        return
    path = _get_scores_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for t, s in (data.get("scores") or {}).items():
                try:
                    _scores[t.strip()] = float(s)
                except (TypeError, ValueError):
                    pass
            for t, c in (data.get("touches") or {}).items():
                try:
                    _touches[t.strip()] = int(c)
                except (TypeError, ValueError):
                    pass
            for t, pts in (data.get("history") or {}).items():
                if isinstance(pts, list):
                    _history[t.strip()] = [[int(r), float(s)] for r, s in pts]
            if isinstance(data.get("meta"), dict):
                _meta.update(data["meta"])
        except Exception as e:
            print(f"[prompt-evolution] Could not read tag_scores.json: {e}")
        _loaded = True
        return

    # Migrate legacy tab-separated .txt, if present
    legacy = _legacy_txt_path()
    if os.path.exists(legacy):
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "\t" not in line:
                        continue
                    tag, _, raw = line.partition("\t")
                    try:
                        _scores[tag.strip()] = float(raw.strip())
                    except ValueError:
                        pass
            _loaded = True
            _flush()  # write the JSON version going forward
            print(f"[prompt-evolution] Migrated {len(_scores)} tags from tag_scores.txt → tag_scores.json")
            return
        except Exception as e:
            print(f"[prompt-evolution] Legacy migration failed: {e}")

    _loaded = True


def _flush() -> None:
    path = _get_scores_path()
    data = {
        "version": 2,
        "scores": _scores,
        "touches": _touches,
        "history": _history,
        "meta": _meta,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=0)
    os.replace(tmp, path)  # atomic-ish write


def _touch(tags: set[str], n: int = 1) -> None:
    for t in tags:
        _touches[t] = _touches.get(t, 0) + n


# ---------------------------------------------------------------------------
# Public API — scoring during play
# ---------------------------------------------------------------------------

def _snapshot(round_idx: int) -> None:
    """Record a sparse history point for any tag whose score changed."""
    for tag, score in _scores.items():
        pts = _history.setdefault(tag, [])
        sc = round(float(score), 4)
        if pts and pts[-1][1] == sc:
            continue  # unchanged → stay sparse
        pts.append([round_idx, sc])
        if len(pts) > _HISTORY_MAX_POINTS:
            del pts[0]


def end_round(decay: Optional[float] = None) -> None:
    """
    Close out a game round: optionally decay all scores toward zero, advance the
    round counter, and snapshot history. `decay` defaults to the stored setting
    (1.0 = off). A factor like 0.98 fades stale preferences over time.
    """
    with _lock:
        _load()
        factor = _meta.get("decay", 1.0) if decay is None else decay
        if factor and factor != 1.0:
            for t in list(_scores.keys()):
                _scores[t] = _scores[t] * factor
        _meta["rounds"] = _meta.get("rounds", 0) + 1
        _snapshot(_meta["rounds"])
        _flush()


def expand_new_tags(prompt: str, parent_prompt: str,
                    exclude: set[str] | None = None) -> set[str]:
    """
    Expanded (colour/part) tags present in `prompt` but NOT in `parent_prompt`.
    e.g. parent 'red eyes, hat' vs 'blue eyes, hat, ocean' -> {'blue', 'ocean'}
    ('eyes' carried over; 'red' was dropped). `exclude` removes e.g. starting tokens.
    """
    def parse(p: str) -> set[str]:
        return {t.strip().lower() for t in p.split(",") if t.strip()}

    win = set(_expand_tags(parse(prompt), 1.0).keys())
    par = set(_expand_tags(parse(parent_prompt), 1.0).keys())
    new = win - par
    if exclude:
        new -= {t.strip().lower() for t in exclude}
    return new


def record_win(winner_prompt: str, parent_prompt: str, weight: float = 1.0,
               exclude: set[str] | None = None) -> int:
    """
    Credit the tags the winner ADDED relative to its parent generation, splitting
    one unit of credit (`weight`) evenly across them: 1 new tag -> full weight,
    k new tags -> weight/k each. Returns how many new tags were credited.
    """
    new = expand_new_tags(winner_prompt, parent_prompt, exclude)
    if not new:
        return 0
    per = weight / len(new)
    with _lock:
        _load()
        for tag in new:
            _scores[tag] = _scores.get(tag, 0) + per
        _touch(new)
        _flush()
    return len(new)
    """Apply `delta` to ALL tags in `prompt` (plus `extra_tags`). Like/dislike buttons."""
    def parse(p: str) -> set[str]:
        return {t.strip().lower() for t in p.split(",") if t.strip()}

    tags = parse(prompt)
    if extra_tags:
        tags |= {t.strip().lower() for t in extra_tags if t.strip()}

    with _lock:
        _load()
        for tag in tags:
            _scores[tag] = _scores.get(tag, 0) + delta
        _touch(tags)
        _flush()


def adjust_tags_set(tags: set[str], delta: float) -> None:
    """Apply `delta` to a pre-built set, expanding colour+part compounds."""
    if not tags:
        return
    with _lock:
        _load()
        expanded = _expand_tags(tags, delta)
        for tag, d in expanded.items():
            _scores[tag] = _scores.get(tag, 0) + d
        _touch(expanded.keys())
        _flush()


# ---------------------------------------------------------------------------
# Public API — manual editing (graph view)
# ---------------------------------------------------------------------------

def set_score(tag: str, value: float) -> None:
    """Set a single tag's score to an exact value."""
    tag = tag.strip().lower()
    if not tag:
        return
    with _lock:
        _load()
        _scores[tag] = float(value)
        _touches.setdefault(tag, 0)
        _flush()


def bump_tag(tag: str, delta: float) -> None:
    """Nudge a single tag's score by `delta`."""
    tag = tag.strip().lower()
    if not tag:
        return
    with _lock:
        _load()
        _scores[tag] = _scores.get(tag, 0) + float(delta)
        _touches.setdefault(tag, 0)
        _flush()


def delete_tag(tag: str) -> None:
    """Remove a tag entirely."""
    tag = tag.strip().lower()
    with _lock:
        _load()
        _scores.pop(tag, None)
        _touches.pop(tag, None)
        _history.pop(tag, None)
        _flush()


def set_many(mapping: dict[str, float], replace: bool = False) -> None:
    """
    Apply a {tag: score} mapping. If `replace` is True, the mapping becomes the
    complete score set (any tag not present is removed) — used by the table editor.
    A score of None deletes that tag.
    """
    cleaned = {t.strip().lower(): v for t, v in mapping.items() if t and t.strip()}
    with _lock:
        _load()
        if replace:
            for old in list(_scores.keys()):
                if old not in cleaned:
                    _scores.pop(old, None)
                    _touches.pop(old, None)
                    _history.pop(old, None)
        for tag, v in cleaned.items():
            if v is None:
                _scores.pop(tag, None)
                _touches.pop(tag, None)
                _history.pop(tag, None)
            else:
                _scores[tag] = float(v)
                _touches.setdefault(tag, 0)
        _flush()


# ---------------------------------------------------------------------------
# Public API — reads
# ---------------------------------------------------------------------------

def get_scores() -> dict[str, float]:
    with _lock:
        _load()
        return dict(_scores)


def get_touches() -> dict[str, int]:
    with _lock:
        _load()
        return dict(_touches)


def get_meta() -> dict:
    with _lock:
        _load()
        return dict(_meta)


def get_history(tags: Optional[list] = None) -> dict:
    """
    Return sparse score history: {tag: [[round, score], ...]}.
    If `tags` is given, only those tags are returned.
    """
    with _lock:
        _load()
        if tags is None:
            return {t: list(p) for t, p in _history.items()}
        out = {}
        for t in tags:
            t = t.strip().lower()
            if t in _history:
                out[t] = list(_history[t])
        return out


def get_decay() -> float:
    with _lock:
        _load()
        return float(_meta.get("decay", 1.0))


def set_decay(factor: float) -> None:
    """Set the per-round decay factor (1.0 = off; e.g. 0.98 fades old scores)."""
    with _lock:
        _load()
        _meta["decay"] = float(factor)
        _flush()


def get_confidence_k() -> float:
    with _lock:
        _load()
        return float(_meta.get("confidence_k", 5.0))


def set_confidence_k(k: float) -> None:
    """Set the confidence damping constant used by weighted_pool (0 = off)."""
    with _lock:
        _load()
        _meta["confidence_k"] = float(k)
        _flush()


def get_setting(name: str, default: float) -> float:
    with _lock:
        _load()
        return float(_meta.get(name, default))


def set_setting(name: str, value: float) -> None:
    with _lock:
        _load()
        _meta[name] = float(value)
        _flush()


def get_swap_penalty() -> float:
    with _lock:
        _load()
        return float(_meta.get("swap_penalty", 0.5))


def set_swap_penalty(v: float) -> None:
    """Penalty (total, diluted across tags) for tags the winner swapped out."""
    with _lock:
        _load()
        _meta["swap_penalty"] = float(v)
        _flush()


def get_pick_penalty() -> float:
    with _lock:
        _load()
        return float(_meta.get("pick_penalty", 0.5))


def set_pick_penalty(v: float) -> None:
    """Penalty (total, diluted across tags) for losing mutations not picked."""
    with _lock:
        _load()
        _meta["pick_penalty"] = float(v)
        _flush()


def top_n(n: int = 10) -> list[tuple[str, float]]:
    scores = get_scores()
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def bottom_n(n: int = 10) -> list[tuple[str, float]]:
    scores = get_scores()
    return sorted(scores.items(), key=lambda x: x[1])[:n]


def scores_path() -> str:
    return _get_scores_path()


def clear_scores() -> None:
    """Wipe all tag scores (and history) from memory and disk."""
    global _scores, _touches, _history, _meta, _loaded
    with _lock:
        decay = _meta.get("decay", 1.0)
        _scores = {}
        _touches = {}
        _history = {}
        _meta = {"rounds": 0, "decay": decay}
        _loaded = True
        _flush()


def prune_zeros(threshold: float = 0.0) -> int:
    """
    Remove tags whose |score| <= threshold (default: exactly-zero noise, e.g.
    tags that always appeared in both winner and loser). Returns count removed.
    """
    with _lock:
        _load()
        dead = [t for t, s in _scores.items() if abs(s) <= threshold]
        for t in dead:
            _scores.pop(t, None)
            _touches.pop(t, None)
            _history.pop(t, None)
        if dead:
            _flush()
        return len(dead)

def weighted_pool(top_boost: int = 10, boost_mult: float = 3.0,
                  confidence_k: float = 5.0) -> tuple[list[str], list[float], set[str]]:
    """
    Build a biased mutation pool (positively-scored tags weighted, top N boosted;
    negatively-scored tags returned as bad_tags to swap out).

    `confidence_k` damps low-evidence tags: a tag's weight is multiplied by
    touches / (touches + k), so a +5 tag seen once contributes far less than a
    +5 tag seen 50 times. Set confidence_k <= 0 to disable (raw score weights).
    """
    scores = get_scores()
    touches = get_touches()
    positive = {t: s for t, s in scores.items() if s > 0}
    negative = {t for t, s in scores.items() if s < 0}

    def confidence(tag: str) -> float:
        if confidence_k <= 0:
            return 1.0
        n = touches.get(tag, 0)
        return n / (n + confidence_k)

    positive_colours = {t: s for t, s in positive.items() if t in _COLORS}
    positive_parts = {t: s for t, s in positive.items() if t in _PARTS}
    compounds: dict[str, float] = {}
    compound_conf: dict[str, float] = {}
    for colour, cscore in positive_colours.items():
        for part, pscore in positive_parts.items():
            name = f"{colour} {part}"
            compounds[name] = min(cscore, pscore)
            # both components must be well-evidenced → use the weaker one
            compound_conf[name] = min(confidence(colour), confidence(part))

    combined = {**positive, **compounds}

    pool_tags, pool_weights = [], []
    if combined:
        top_tags = set(t for t, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_boost])
        for tag, score in combined.items():
            conf = compound_conf.get(tag, confidence(tag))
            w = float(score) * conf
            if tag in top_tags:
                w *= boost_mult
            if w > 0:
                pool_tags.append(tag)
                pool_weights.append(w)

    bad_compounds = {
        f"{c} {p}"
        for c in _COLORS for p in _PARTS
        if scores.get(c, 0) < 0 or scores.get(p, 0) < 0
    }
    bad_tags = negative | bad_compounds

    return pool_tags, pool_weights, bad_tags
