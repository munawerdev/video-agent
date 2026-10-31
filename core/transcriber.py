import os

import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


_model = None


def load_model():

    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str, language: str | None = None) -> str:

    model = load_model()

    options = {"task": "transcribe"}
    if language is not None:
        options["language"] = language
    result = model.transcribe(chunk_path, **options)
    return result["text"]


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Transcribe one chunk with Whisper, using a language hint when recognized.
    Hindi and Hinglish are transcribed in their original language.
    """
    language_hint = {
        "english": "en",
        "hindi": "hi",
        "hinglish": "hi",
    }.get(language.lower())
    return transcribe_chunk_whisper(chunk_path, language=language_hint)


def transcribe_all(chunks: list, language: str = "english") -> str:

    full_transcript = ""

    print("Using Whisper for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()
