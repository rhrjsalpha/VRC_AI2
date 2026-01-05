# Speak_voice.py  (Coqui XTTS v2 - speaker_wav REQUIRED, no file, stoppable playback)
# pip install TTS sounddevice numpy
import os
import threading
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd

from TTS.api import TTS

# =========================
# Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_OUT_DEVICE_KEYWORD = "INZONE H9 / INZONE H7 - Chat"
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "ko"

# ✅ 여기에 ref.wav 경로를 꼭 넣어줘 (10~20초 권장)
DEFAULT_SPEAKER_WAV = r"E:\VRC_AI2\LTS_MODULE\wavs\JeongUnHaye.wav"


# =========================
# Device utils
# =========================
def find_output_device(keyword: str):
    if not keyword:
        return None
    keyword = keyword.lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_output_channels", 0) > 0 and keyword in dev["name"].lower():
            return i
    return None


# =========================
# Coqui loader (load once)
# =========================
_tts_lock = threading.Lock()
_tts_obj: Optional[TTS] = None

def _load_tts_once() -> TTS:
    global _tts_obj
    if _tts_obj is not None:
        return _tts_obj

    with _tts_lock:
        if _tts_obj is not None:
            return _tts_obj

        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

        print(f"[XTTS] loading model on GPU? {use_gpu}", flush=True)
        _tts_obj = TTS(model_name=XTTS_MODEL_NAME, gpu=use_gpu)  # ✅ 핵심
        return _tts_obj


# =========================
# Synthesize
# =========================
def synthesize_xtts_wav(
    text: str,
    *,
    language: str = DEFAULT_LANGUAGE,
    speaker_wav: str = DEFAULT_SPEAKER_WAV,
) -> Tuple[np.ndarray, int]:
    """
    speaker_wav 기반 합성 (XTTS v2에서는 이 방식이 제일 안정적)
    return: (wav_float32_mono, sample_rate=24000)
    """
    text = (text or "").strip()
    if not text:
        return np.zeros((0,), dtype=np.float32), 24000

    speaker_wav = (speaker_wav or "").strip()
    if not speaker_wav:
        raise RuntimeError("[XTTS] speaker_wav is required for this setup/model.")
    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f"[XTTS] speaker_wav not found: {speaker_wav}")

    tts = _load_tts_once()
    sr = 24000

    wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
    wav = np.asarray(wav, dtype=np.float32).flatten()

    if wav.size == 0:
        raise RuntimeError("[XTTS] synthesized zero samples (empty wav).")

    wav = np.clip(wav, -1.0, 1.0)
    return wav, sr


# =========================
# Playback (stoppable)
# =========================
def play_wav_float32(
    wav: np.ndarray,
    sr: int,
    *,
    out_device_keyword: str = DEFAULT_OUT_DEVICE_KEYWORD,
    volume: float = 1.0,
    stop_event: Optional[threading.Event] = None,
):
    if wav is None or wav.size == 0:
        return

    dev_id = find_output_device(out_device_keyword)
    if out_device_keyword and dev_id is None:
        raise RuntimeError(f"Output device not found: {out_device_keyword}")

    x = wav.astype(np.float32, copy=False)
    if volume != 1.0:
        x = np.clip(x * float(volume), -1.0, 1.0)

    idx = 0
    n = x.shape[0]

    def callback(outdata, frames, time, status):
        nonlocal idx
        if stop_event is not None and stop_event.is_set():
            raise sd.CallbackStop()

        end = idx + frames
        chunk = x[idx:end]
        if chunk.shape[0] < frames:
            outdata[:chunk.shape[0], 0] = chunk
            outdata[chunk.shape[0]:, 0] = 0.0
            raise sd.CallbackStop()
        else:
            outdata[:, 0] = chunk
            idx = end

    with sd.OutputStream(
        samplerate=int(sr),
        channels=1,
        dtype="float32",
        device=dev_id,
        callback=callback,
    ):
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            sd.sleep(50)
            if idx >= n:
                break


# =========================
# Public API (keep compatibility)
# =========================
def speak_async(
    text: str,
    out_device_keyword: str = DEFAULT_OUT_DEVICE_KEYWORD,
    *,
    volume: float = 1.0,
    stop_event: Optional[threading.Event] = None,
    language: str = DEFAULT_LANGUAGE,
    speaker_wav: str = DEFAULT_SPEAKER_WAV,
):
    if stop_event is None:
        stop_event = threading.Event()

    def _worker():
        try:
            wav, sr = synthesize_xtts_wav(text, language=language, speaker_wav=speaker_wav)
            print(f"[XTTS] sr={sr} frames={wav.size} peak={float(np.max(np.abs(wav))):.3f}", flush=True)
            play_wav_float32(
                wav, sr,
                out_device_keyword=out_device_keyword,
                volume=volume,
                stop_event=stop_event,
            )
        except Exception as e:
            print("[Speak_voice] ERROR:", repr(e), flush=True)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    return th, stop_event


# =========================
# Self test
# =========================
if __name__ == "__main__":
    print("=== SELF TEST: Coqui XTTS Speak_voice.py (speaker_wav mode) ===")
    print("ref wav:", DEFAULT_SPEAKER_WAV, "exists?", os.path.exists(DEFAULT_SPEAKER_WAV))

    print("\n--- Output devices containing 'inzone' ---")
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_output_channels", 0) > 0 and "inzone" in dev["name"].lower():
            print(i, dev["name"])

    kw = "INZONE H9 / INZONE H7 - Chat"
    th, st = speak_async(
        "스피커 참조 음성으로 테스트 중입니다. 들리면 성공입니다.",
        out_device_keyword=kw,
        volume=1.0,
    )
    th.join()
    print("Done (thread alive?):", th.is_alive())
