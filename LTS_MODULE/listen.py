import numpy as np
import threading
import sounddevice as sd
from faster_whisper import WhisperModel
import time
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

def record_until_silence(MIC_ID, FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR):
    # RMS_TH = calibrate_noise(MIC_ID, SR)
    # print("Auto RMS_TH:", RMS_TH)
    RMS_TH = 0.03
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

    if not recorded:
        return None, "no_audio"

    audio = np.concatenate(recorded, axis=0).astype(np.float32)
    return audio, reason["v"] or "unknown"

def SST_Wisper_Model(MIC_device_name, FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR):
    MIC_ID = find_input_device(MIC_device_name)
    print("Using mic device:", MIC_ID)

    rec_start = time.time()
    audio, reason = record_until_silence(MIC_ID, FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR)  # ★ 언패킹
    rec_end = time.time()

    if audio is None or audio.size == 0:
        print(f"No speech detected. reason={reason}, rec_total: {rec_end - rec_start:.4f}s")
        return

    print(f"Recorded {audio.size / SR:.2f}s, rec_total: {rec_end - rec_start:.4f}s, reason={reason}")

    model_start = time.time()
    model = model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=1,
)
    segments, info = model.transcribe(audio, language="ko", vad_filter=True)
    model_end = time.time()

    text = " ".join(seg.text for seg in segments).strip()
    print("STT:", text, "time:", model_end - model_start)
    return text

# ===== 실행 =====
if __name__ == "__main__":
    SR = 16000
    # ---- VAD(무음 감지) 파라미터 ----
    FRAME_MS = 30  # 프레임 길이(ms) 20~30 권장
    FRAME = int(SR * FRAME_MS / 1000)
    START_SEC = 0.15  # 이만큼 연속으로 소리나면 "말 시작"
    END_SEC = 0.60  # 이만큼 연속 무음이면 "말 종료"
    MAX_SEC = 10.0  # 최대 녹음 길이(안전장치)

    SST_Wisper_Model("INZONE", FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR)



