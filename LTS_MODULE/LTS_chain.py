# LTS_chain.py
import time, queue, threading
from datetime import datetime
from typing import List

from thinking import model_thinking
from Speak_text import send_chatbox, set_typing
from Speak_voice import speak_async

# ✅ 여기서는 "새로 만든 recorder_thread/stt_thread"를 import한다고 가정
#    (네 listen.py에 넣거나, 별도 모듈로 빼도 됨)
from listen import recorder_thread, stt_thread, find_input_device
from faster_whisper import WhisperModel

import edge_tts
print(edge_tts.__version__)

FRAME_MS = 25
SR = 16000
FRAME = int(SR * FRAME_MS / 1000)

START_SEC = 1
END_SEC   = 3
MAX_SEC   = 10

MIC_DEVICE = "Voicemeeter Out B1" # 내 마이크    # 프로그램소리 : Voicemeeter Out B1 # 내 마이크 : INZONE

MAX_KEEP_UTTERANCES = 5

def send_chatbox_long(text: str, *, chunk_size: int = 144, delay: float = 0.35):
    """
    Speak_text.send_chatbox()는 144자로 잘라버리므로,
    여기서 미리 144자 단위로 쪼개서 여러 번 보내기.
    """
    text = (text or "").strip()
    if not text:
        return

    parts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    for p in parts:
        send_chatbox(p, immediate=True, notify=False)
        if len(parts) > 1:
            time.sleep(delay)

def drain_merge_latest(
    text_q: "queue.Queue",
    *,
    max_keep: int = 3,
    prefix: str = "- "
) -> str:
    """
    text_q에 쌓인 텍스트를 전부 꺼내서,
    가장 최근 max_keep개만 남기고 한 덩어리로 합쳐 반환.
    (최소 1개는 blocking get으로 기다림)
    """
    print(text_q.queue)
    items: List[str] = []

    # 최소 1개는 반드시 받기 (block)
    first = text_q.get()
    t = first if isinstance(first, str) else first[0]
    if t and str(t).strip():
        items.append(str(t).strip())

    # 나머지는 가능한 한 전부 drain (non-block)
    while True:
        try:
            x = text_q.get_nowait()
        except queue.Empty:
            break
        t = x if isinstance(x, str) else x[0]
        if t and str(t).strip():
            items.append(str(t).strip())

    # 최근 max_keep개만
    if not items:
        return ""

    tail = items[-max_keep:]

    # 합치기
    if len(tail) == 1:
        return tail[0]
    return "\n".join(prefix + s for s in tail)

def main_loop():
    # 1) 모델 1회 로드 (매 호출 로드 금지)
    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4, num_workers=1)

    # 2) 큐 2개
    audio_q = queue.Queue(maxsize=8)    # 녹음 청크
    text_q  = queue.Queue(maxsize=200)  # STT 텍스트

    # 3) stop 이벤트
    stop_rec = threading.Event()
    stop_stt = threading.Event()

    # 4) 마이크 id
    MIC_ID = find_input_device(MIC_DEVICE)
    print("Using mic device:", MIC_ID)

    utter_event = threading.Event()
    stt_busy_event = threading.Event()

    # 5) 스레드 시작
    t_rec = threading.Thread(
        target=recorder_thread,
        args=(audio_q, MIC_ID, FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR, stop_rec, utter_event),
        daemon=True
    )
    t_stt = threading.Thread(
        target=stt_thread,
        args=(audio_q, text_q, model, stop_stt, stt_busy_event),
        daemon=True
    )
    t_rec.start()
    t_stt.start()
    print("[INFO] recorder/stt threads started.")


    try:
        tts_th = None
        tts_stop = None

        while True:
            # (A) TTS가 말하는 중이면 끝까지 기다림
            if tts_th is not None and tts_th.is_alive():
                tts_th.join()

            # (B) 여기 핵심: "현재 발화"가 끝날 때까지 기다림
            #  - 사용자가 말하고 있으면(녹음 중) silence로 끝날 때까지 기다림
            while utter_event.is_set():
                time.sleep(0.02)

            #  - STT 변환이 진행 중이면 끝날 때까지 기다림
            while stt_busy_event.is_set():
                time.sleep(0.02)

            # (C) 이제 text_q에 쌓인 것들을 한 번에 드레인 → 최근 n개 합치기

            merged = drain_merge_latest(text_q, max_keep=MAX_KEEP_UTTERANCES)

            print(merged)
            if not merged:
                continue

            ask = (
                f"{merged}"
            )

            say_text = model_thinking("lumi-deepseek", ask=ask)
            print(say_text)
            set_typing(True)
            send_chatbox_long(f"AI_Bot: {say_text}")  # <= 144자 넘으면 자동 분할
            set_typing(False)

            # (D) TTS 시작
            tts_th, tts_stop = speak_async(say_text, out_device_keyword="Cable Input")

    except KeyboardInterrupt:
        print("Stopping...")
        stop_rec.set()
        stop_stt.set()
        time.sleep(0.2)

if __name__ == "__main__":
    main_loop()


 # "gemma3:1b" "lumi"
# 설정 : windows 기본장치 : Voicemeter Input(VB-Audio)