"""Generic experiment-config validation.

Catches the mistakes that are cheap to catch before burning GPU time:
missing required keys, unknown keys (typos), and bad enum values. Anything
project-specific (dataset paths existing, a model name being registered,
a template file existing) is supplied via ``extra_checks``.
"""

from typing import Callable, Iterable


def validate_config(
    config: dict,
    default_config: dict,
    required_keys: Iterable[str] = (),
    allowed_values: dict[str, set] | None = None,
    extra_checks: Iterable[Callable[[dict], str | None]] = (),
    exempt_unknown_keys: Iterable[str] = (),
) -> None:
    """Sanity-check a merged experiment config.

    Raises ValueError listing every problem found (not just the first).

    Args:
        config: the merged experiment config (defaults + per-experiment overrides).
        default_config: the complete set of keys the codebase reads — anything
            in ``config`` outside this set (and ``required_keys``/
            ``exempt_unknown_keys``) is flagged as a likely typo.
        required_keys: keys that must be present and truthy.
        allowed_values: {key: set of allowed values} for enum-like keys.
        extra_checks: callables taking the config and returning an error
            string (or None if OK) — for project-specific checks like
            "does this dataset path exist" or "is this model registered".
        exempt_unknown_keys: keys allowed outside ``default_config`` without
            being flagged (e.g. a runtime-injected job-id suffix, or a
            JSON-doesn't-support-comments convention key).
    """
    errors = []

    for key in required_keys:
        if not config.get(key):
            errors.append(f"missing required key: {key!r}")

    unknown = set(config) - set(default_config) - set(required_keys) - set(exempt_unknown_keys)
    if unknown:
        errors.append(f"unknown key(s) not in default_config (typo?): {sorted(unknown)}")

    for key, allowed in (allowed_values or {}).items():
        value = config.get(key)
        if value is not None and value not in allowed:
            errors.append(f"{key}={value!r} not in {sorted(allowed)}")

    for check in extra_checks:
        error = check(config)
        if error:
            errors.append(error)

    if errors:
        name = config.get("name", "<unnamed>")
        raise ValueError(f"Invalid config {name!r}:\n  " + "\n  ".join(errors))
