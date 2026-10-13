#!/usr/bin/env python3
"""
console.py — terminal output policy for the museum guide.

The pipeline prints two very different kinds of line. A handful ARE the demo:
what the visitor said, what the robot answered, what it detected on the face,
which painting it picked and how confident it was. The rest is engineering
telemetry: viewer ports, TTS backend handshakes, gesture-script builds, per-
waypoint walk progress, triple counts, saved-file paths. Interleaved, the
second kind buries the first — which is exactly what makes a screen recording
of the terminal unreadable.

So every runtime print goes through one of these:

    event(...)   — always shown. The demo narrative + anything the operator
                   must see (warnings, errors) to know something went wrong.
    detail(...)  — shown only in verbose mode. Diagnostics.
    warn/error() — always shown, prefixed.

VERBOSE IS THE DEFAULT: nothing changes for day-to-day debugging or for the
benchmark scripts. Quiet mode is opt-in, per run:

    python3 face_trigger.py --webcam --quiet
    python3 guida_museale_core.py --quiet
    MUSEUM_QUIET=1 python3 ...            # same thing, for indirect callers

Deliberately not the `logging` module: these lines are a user-facing
transcript (emoji, inline `end=""` progress, aligned indentation), not log
records — levels/handlers/formatters would fight that, and half the call sites
stream partial lines while the TTS speaks.
"""

import contextlib
import os
import sys

_QUIET: bool = os.environ.get("MUSEUM_QUIET", "").strip().lower() in ("1", "true", "yes", "on")


def set_quiet(quiet: bool) -> None:
    """Called once from main() after parsing --quiet. Also exported to the
    environment so an in-process module imported later (and any subprocess we
    spawn) picks up the same policy."""
    global _QUIET
    _QUIET = bool(quiet)
    os.environ["MUSEUM_QUIET"] = "1" if quiet else "0"


def is_quiet() -> bool:
    return _QUIET


def event(*args, **kwargs) -> None:
    """A line that belongs in the demo transcript. Always printed."""
    print(*args, **kwargs)


def detail(*args, **kwargs) -> None:
    """Diagnostic chatter. Printed only when NOT in quiet mode."""
    if not _QUIET:
        print(*args, **kwargs)


def warn(*args, **kwargs) -> None:
    """Something the operator should see even mid-demo (degraded mode,
    fallback taken, mismatch). Always printed."""
    print(*args, **kwargs)


def error(*args, **kwargs) -> None:
    """A failure. Always printed."""
    print(*args, **kwargs)


@contextlib.contextmanager
def muted_stderr():
    """
    Silence fd 2 for the duration of the block — ONLY used around the PyAudio/
    ALSA microphone handshake, which dumps a screenful of
    `ALSA lib pcm_*.c: Unknown PCM cards...` on most Linux boxes before every
    single utterance. Those come from the C library, so Python-level
    redirection doesn't catch them; the file descriptor has to be swapped.

    No-op outside quiet mode, so ordinary debugging still sees everything.
    """
    if not _QUIET:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        sys.stderr.flush()
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
