"""Diagnostic output is opt-in; warnings, results and progress keep using print."""
import os


def diagnostic_print(*args, **kwargs):
    """Use STCKLA_VERBOSE=1 to restore the original detailed diagnostic output."""
    if os.environ.get("STCKLA_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(*args, **kwargs)
