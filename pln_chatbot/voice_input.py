"""
Entrada por voz para la consola.

Convierte audio del micrófono a texto con SpeechRecognition y devuelve una
cadena que luego usa el mismo pipeline de PLN que las entradas escritas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceResult:
    text: str | None
    error: str | None = None


def listen_once(language: str = "es-MX", timeout: int = 5, phrase_time_limit: int = 8) -> VoiceResult:
    try:
        import speech_recognition as sr
    except ImportError:
        return VoiceResult(
            None,
            "No está instalada la librería SpeechRecognition. Instale con: pip install SpeechRecognition",
        )

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Escuchando... hable ahora.")
            recognizer.adjust_for_ambient_noise(source, duration=0.6)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except AttributeError:
        return VoiceResult(
            None,
            "No se pudo acceder al micrófono. En macOS puede requerir PyAudio y permisos de micrófono.",
        )
    except OSError as exc:
        return VoiceResult(
            None,
            f"No se encontró un micrófono disponible o falta soporte de audio: {exc}",
        )
    except sr.WaitTimeoutError:
        return VoiceResult(None, "No se detectó voz dentro del tiempo de espera.")

    try:
        text = recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return VoiceResult(None, "No pude entender el audio. Intente hablar más claro o más cerca del micrófono.")
    except sr.RequestError as exc:
        return VoiceResult(None, f"No se pudo usar el servicio de reconocimiento de voz: {exc}")

    return VoiceResult(text.strip(), None)
