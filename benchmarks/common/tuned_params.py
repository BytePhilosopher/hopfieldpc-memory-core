"""Store of tuned hyperparameters, one JSON file per dataset.

Each dataset folder (``benchmarks/<dataset>/``) has a ``tuning.py`` that is the
only code allowed to tune anything, and a ``tuned_params.json`` next to it that
holds the results. Baseline and Hopfield experiments only read that file, so
tuning runs once per dataset and settings and is never repeated by an
experiment. The file is meant to be committed.

Layout::

    {
      "<profile>": {             # "full" or "quick" (the --quick smoke settings)
        "<section>": {           # e.g. "baseline", "hopfield_l8", "hopfield_l5"
          "fingerprint": {...},  # every setting the tuned values depend on
          "values": {...},       # the tuned values (and the curves behind them)
          "tuned_on": "YYYY-MM-DD"
        }
      }
    }

An entry is reused only when its fingerprint equals the one computed from the
current config, so changing a relevant setting makes the stored values stale:
experiments then stop with a message to rerun ``tuning.py``, and ``tuning.py``
retunes only the stale sections.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

FULL = "full"
QUICK = "quick"


class TunedParamsError(RuntimeError):
    """Raised when an experiment needs tuned values that are missing or stale."""


def profile_name(quick: bool) -> str:
    return QUICK if quick else FULL


def normalize(obj: Any) -> Any:
    """JSON round trip, so tuples become lists and fingerprints compare like stored ones."""
    return json.loads(json.dumps(obj))


def load_store(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        store = json.load(f)
    if not isinstance(store, dict):
        raise TypeError(f"{path} must hold a JSON object, got {type(store).__name__}.")
    return store


def get_entry(path: str | Path, profile: str, section: str) -> dict[str, Any] | None:
    return load_store(path).get(profile, {}).get(section)


def lookup(path: str | Path, profile: str, section: str, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    """Stored values for ``section`` if they were tuned with exactly these settings, else None."""
    entry = get_entry(path, profile, section)
    if entry is None or entry.get("fingerprint") != normalize(fingerprint):
        return None
    return entry["values"]


def save_entry(
    path: str | Path,
    profile: str,
    section: str,
    fingerprint: dict[str, Any],
    values: dict[str, Any],
) -> None:
    """Write one section, keeping every other entry; atomic so an interrupted write loses nothing."""
    path = Path(path)
    store = load_store(path)
    store.setdefault(profile, {})[section] = {
        "fingerprint": normalize(fingerprint),
        "values": normalize(values),
        "tuned_on": datetime.date.today().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(
    path: str | Path,
    profile: str,
    section: str,
    fingerprint: dict[str, Any],
    *,
    tuning_command: str,
) -> dict[str, Any]:
    """Stored values for ``section``; raise with the command to run when missing or stale."""
    values = lookup(path, profile, section, fingerprint)
    if values is not None:
        return values
    entry = get_entry(path, profile, section)
    if entry is None:
        reason = f"no tuned '{section}' parameters ({profile} profile) in {path}"
    else:
        stored = entry.get("fingerprint", {})
        wanted = normalize(fingerprint)
        changed = sorted(k for k in set(stored) | set(wanted) if stored.get(k) != wanted.get(k))
        reason = (f"tuned '{section}' parameters ({profile} profile) in {path} were produced with "
                  f"different settings (changed: {', '.join(changed)})")
    raise TunedParamsError(f"{reason}. Run `{tuning_command}` first.")
