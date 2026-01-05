import numpy as np
import threading
import sounddevice as sd
from faster_whisper import WhisperModel
import time, queue
# ===== 마이크 선택 =====
def find_input_device(keyword):
    """select mic device"""
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and keyword.lower() in dev["name"].lower():
            return i
    return None

def rms(x: np.ndarray) -> float:
    """음량 계산 함수"""
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def calibrate_noise(MIC_ID, SR, seconds=0.5):
    x = sd.rec(int(SR*seconds), samplerate=SR, channels=1, dtype=np.float32, device=MIC_ID)
    sd.wait()
    x = x.flatten()
    noise = float(np.sqrt(np.mean(x*x) + 1e-12))
    # 배경소음의 3~5배 정도를 임계값으로
    return max(0.003, noise * 3.0)

def record_until_silence(MIC_ID, FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR, utter_event=None):
    # RMS_TH = calibrate_noise(MIC_ID, SR)
    # print("Auto RMS_TH:", RMS_TH)
    RMS_TH = 0.01
    START_FRAMES  = int(START_SEC * 1000 / FRAME_MS)
    END_FRAMES    = int(END_SEC   * 1000 / FRAME_MS)
    MAX_FRAMES    = int(MAX_SEC   * 1000 / FRAME_MS)
    LEADIN_FRAMES = START_FRAMES

    ring, recorded = [], []
    started = False
    voiced_run = 0
    silent_run = 0

    done = threading.Event()
    reason = {"v": None}

    def rms(x):
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def callback(indata, frames, time_info, status):
        nonlocal started, voiced_run, silent_run, ring, recorded

        x = indata[:, 0].copy()
        r = rms(x)

        ring.append(x)
        if len(ring) > LEADIN_FRAMES:
            ring.pop(0)

        is_voiced = (r >= RMS_TH)

        if not started:
            voiced_run = voiced_run + 1 if is_voiced else 0
            if voiced_run >= START_FRAMES:
                started = True
                if utter_event is not None:
                    utter_event.set()
                recorded.extend(ring)
                ring.clear()
                silent_run = 0
        else:
            recorded.append(x)
            silent_run = 0 if is_voiced else (silent_run + 1)

            if silent_run >= END_FRAMES:
                reason["v"] = "silence"
                done.set()
                raise sd.CallbackStop()

            if len(recorded) >= MAX_FRAMES:
                reason["v"] = "maxlen"
                done.set()
                raise sd.CallbackStop()

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        dtype="float32",
        blocksize=FRAME,
        device=MIC_ID,
        callback=callback,
    ):
        # 종료 신호가 오거나, 최악의 경우 timeout
        done.wait(timeout=MAX_SEC + 2.0)

    if utter_event is not None:
        utter_event.clear()  # 발화 캡처 종료(무음/timeout 포함)

    if not recorded:
        return None, "no_audio"

    audio = np.concatenate(recorded, axis=0).astype(np.float32)
    return audio, reason["v"] or "unknown"

def recorder_thread(audio_q, MIC_ID, FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR, stop_event_recorder,  utter_event=None):
    while not stop_event_recorder.is_set():
        audio, reason = record_until_silence(MIC_ID, FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR, utter_event=utter_event)
        if audio is None:
            continue
        audio_q.put((audio, reason))

def stt_thread(audio_q, text_q, model, stop_event_sst, stt_busy_event=None):
    while not stop_event_sst.is_set():
        try:
            audio, reason = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if stt_busy_event is not None:
            stt_busy_event.set()  # ✅ "지금 STT 중"

        segments, info = model.transcribe(audio, language="ko", vad_filter=True)
        text = " ".join(seg.text for seg in segments).strip()
        if text:
            text_q.put(text)

        if stt_busy_event is not None:
            stt_busy_event.clear()  # ✅ "STT 끝"

def run_chain(MIC_device_name="INZONE"):
    SR = 16000
    FRAME_MS = 30
    FRAME = int(SR * FRAME_MS / 1000)
    START_SEC = 0.15
    END_SEC = 0.60
    MAX_SEC = 10.0

    MIC_ID = find_input_device(MIC_device_name)
    print("Using mic device:", MIC_ID)

    # ★ 핵심: 모델은 한 번만 로드 (가능하면 메인에서)
    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4, num_workers=1)

    audio_q = queue.Queue(maxsize=8)
    text_q  = queue.Queue(maxsize=8)
    stop_event_rec = threading.Event()
    stop_event_sst = threading.Event()

    t_rec = threading.Thread(
        target=recorder_thread,
        args=(audio_q, MIC_ID, FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR, stop_event_rec),
        daemon=False,
    )
    t_stt = threading.Thread(
        target=stt_thread,
        args=(audio_q, text_q, model, stop_event_sst),
        daemon=False,
    )

    t_rec.start()
    t_stt.start()

    try:
        while True:
            print(audio_q.queue)
            print(text_q.queue)
            text = text_q.get()
            print("STT:", text)
            # 여기서 LLM 응답/tts로 이어가면 됨
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event_rec.set()
        stop_event_sst.set()
        t_rec.join()
        t_stt.join()

if __name__ == "__main__":
    run_chain("INZONE")



