import json
import os
import hashlib
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Lightweight prompt versioning -- a scoped-down adaptation of
# trustworthy-ai-eval-architecture.md's Fair-Comparison Controller (section
# 3.2), fitted to what "the prompt" actually is in this project: the
# dataset row's Input text (there is no separate system-prompt/template
# layer here). The full document's "capability vs. prompt-assisted
# condition" machinery is multi-model-comparison scoped and doesn't apply;
# what does apply, and what this delivers, is: knowing WHICH version of a
# prompt produced WHICH result, so run_comparison can answer not just
# "did the result change" but "did my prompt edit cause it."
#
# Passive observability, not a consequential human decision (unlike
# benchmark promotion or calibration judgments) -- recording is automatic,
# happens every time a prompt is built, and never modifies the prompt or
# the dataset. A new version entry is only appended when the text actually
# changed, so re-running an unedited prompt doesn't create history noise.
# ---------------------------------------------------------------------------

_STORE_PATH = os.path.join(os.path.dirname(__file__), "data", "prompt_versions.json")


def _ensure_file():
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    if not os.path.exists(_STORE_PATH):
        with open(_STORE_PATH, "w", encoding="utf-8") as store_file:
            json.dump({}, store_file, indent=2)


def load_prompt_version_history():
    _ensure_file()
    with open(_STORE_PATH, "r", encoding="utf-8") as store_file:
        return json.load(store_file)


def _save_prompt_version_history(history):
    _ensure_file()
    with open(_STORE_PATH, "w", encoding="utf-8") as store_file:
        json.dump(history, store_file, indent=2)


def _hash_text(text):
    # str() rather than `text or ""`: pandas hands back float('nan') for a
    # blank Excel cell, and NaN is truthy, so the `or` fallback is skipped and
    # .encode() is called on a float. That raised AttributeError inside
    # build_prompt, before the generator was ever reached.
    return hashlib.sha256(
        ("" if text is None else str(text)).encode("utf-8")
    ).hexdigest()[:12]


def _diff_summary(previous_text, current_text):

    if previous_text is None:
        return {"change_type": "initial", "length_delta": len(current_text or "")}

    return {
        "change_type": "edited",
        "length_delta": len(current_text or "") - len(previous_text or ""),
        "previous_length": len(previous_text or ""),
        "current_length": len(current_text or ""),
    }


def record_prompt_usage(slot_key, prompt_text, category=None, prompt_assisted=False):
    """Logs the prompt text actually used for a given test-case slot.
    slot_key should use the same identity scheme as
    run_comparison.comparator's default matching (category + position
    within category) so version history lines up with run comparisons.

    prompt_assisted marks whether this specific edit came from an AI
    suggestion (per trustworthy-ai-eval-architecture.md 3.2's
    prompt_assisted: true) rather than a human authoring it directly.
    Defaults to False -- build_prompt's automatic calls never claim a new
    version was AI-assisted on their own; only an explicit call (e.g. a
    human recording it ahead of running the suite) can set this.

    Returns the current version entry for this slot (the newly created one
    if the text changed, otherwise the existing latest one -- including
    its existing prompt_assisted flag, which is never overwritten by a
    later call with the same text)."""

    history = load_prompt_version_history()
    slot_history = history.setdefault(str(slot_key), [])

    previous_text = slot_history[-1]["prompt_text"] if slot_history else None

    if slot_history and previous_text == prompt_text:
        return slot_history[-1]

    entry = {
        "version": len(slot_history) + 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "prompt_text": prompt_text,
        "prompt_hash": _hash_text(prompt_text),
        "diff_from_previous": _diff_summary(previous_text, prompt_text),
        "prompt_assisted": prompt_assisted,
    }

    slot_history.append(entry)
    _save_prompt_version_history(history)

    return entry


def get_prompt_history(slot_key):
    return load_prompt_version_history().get(str(slot_key), [])


def get_latest_version(slot_key):
    slot_history = get_prompt_history(slot_key)
    return slot_history[-1] if slot_history else None
