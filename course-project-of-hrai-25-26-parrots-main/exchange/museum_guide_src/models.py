#!/usr/bin/env python3
"""
Single place where the hosted model ids live.

They used to be repeated as a private constant in each module that called the
API, which meant a provider retiring a model took four separate edits to
recover from — and it has now happened twice: Groq retired
meta-llama/llama-4-scout-17b-16e-instruct (see painting_recognizer.py) and
then llama-3.3-70b-versatile, which the dialogue, the safety gate and the
synthetic survey all named independently.

Both ids can be overridden from the environment, so the next retirement is a
shell variable rather than a patch:

    MUSEUM_CHAT_MODEL=some/other-model pixi run python guida_museale_core.py

available_models() lists what the key can actually reach, which is the one
question worth asking when a call starts failing.
"""
import os
import re

DEFAULT_CHAT_MODEL = "qwen/qwen3.6-27b"
CHAT_MODEL = os.environ.get("MUSEUM_CHAT_MODEL", DEFAULT_CHAT_MODEL)
CHAT_REASONING_EFFORT = os.environ.get("MUSEUM_CHAT_REASONING_EFFORT", "none")

DEFAULT_VISION_MODEL = "qwen/qwen3.6-27b"
VISION_MODEL = os.environ.get("MUSEUM_VISION_MODEL", DEFAULT_VISION_MODEL)

DEFAULT_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
TRANSCRIPTION_MODEL = os.environ.get(
    "MUSEUM_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL
)


def chat_kwargs() -> dict:
    """
    Model-specific keyword arguments every chat call has to pass.

    Kept as a function rather than spread through the call sites so that
    swapping to a non-reasoning model means editing this one place: a model
    that rejects `reasoning_effort` only needs the key dropped here, not in
    five files.
    """
    kwargs = {"model": CHAT_MODEL}
    if CHAT_REASONING_EFFORT:
        kwargs["reasoning_effort"] = CHAT_REASONING_EFFORT
    return kwargs


def available_models(api_key: str | None = None) -> list[str]:
    """
    Model ids the configured API key can actually reach, sorted.

    Returns [] if the listing itself fails (no key, no network): the caller is
    diagnosing a failure already and a second exception on top of the first
    would only bury it.
    """
    try:
        from groq import Groq

        client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY", ""))
        return sorted(m.id for m in client.models.list().data)
    except Exception:
        return []


def explain_model_error(exc: Exception, model: str) -> str:
    """
    One-line diagnosis for a failed chat call.

    A retired model reports as a generic API error whose text mentions
    model_not_found / 404, and the useful reply to that is not the traceback
    but the list of ids that WOULD work — otherwise the next step is always
    the same manual curl.

    A rate limit reports as a ~500-character JSON blob repeating the org id,
    the tier and the billing URL. The operator needs exactly two facts out of
    it — that it was a quota, and when to retry — so it is collapsed to those;
    dumped verbatim it swamps the transcript, and it fires on every call once
    the daily budget runs out.
    """
    text = str(exc)
    if "rate_limit_exceeded" in text or "Rate limit reached" in text:
        which = re.search(r"\b(OTPM|TPM|TPD|RPD|RPM)\b", text)
        retry = re.search(r"try again in ([0-9]+[hms][0-9hms.]*?)\.?(?:\s|$)", text)
        return (f'limite di richieste Groq raggiunto '
                f'({which.group(1) if which else "quota"}) sul modello "{model}"'
                + (f" — riprova tra {retry.group(1)}" if retry else ""))
    if "model_not_found" not in text and "404" not in text:
        return text
    ids = available_models()
    if not ids:
        return (
            f'model "{model}" was rejected as unknown, and the model list could '
            f"not be fetched either — check GROQ_API_KEY and connectivity. ({text})"
        )
    return (
        f'model "{model}" is no longer served by this key. Available: '
        f"{', '.join(ids)}. Set MUSEUM_CHAT_MODEL to one of these."
    )


if __name__ == "__main__":
    print(f"chat model:          {CHAT_MODEL}")
    print(f"  reasoning_effort:  {CHAT_REASONING_EFFORT or '(not sent)'}")
    print(f"vision model:        {VISION_MODEL}")
    print(f"transcription model: {TRANSCRIPTION_MODEL}")
    ids = available_models()
    if not ids:
        print("\nCould not list models — is GROQ_API_KEY set?")
    else:
        print(f"\n{len(ids)} models reachable with this key:")
        for i in ids:
            mark = "  <- in use" if i in (CHAT_MODEL, VISION_MODEL, TRANSCRIPTION_MODEL) else ""
            print(f"  {i}{mark}")
        for label, m in (("chat", CHAT_MODEL), ("vision", VISION_MODEL),
                         ("transcription", TRANSCRIPTION_MODEL)):
            if m not in ids:
                print(f"\n⚠️  the {label} model \"{m}\" is NOT in that list — calls using it will fail.")
