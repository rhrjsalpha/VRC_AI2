# LTS_chain.py (파일 최상단, 다른 import 전에!)
import os
import sys
from pathlib import Path

def add_cuda_dll_dirs():
    # 현재 파이썬 실행 환경(Conda env) 루트
    env = Path(sys.prefix)

    candidates = [
        env / "Library" / "bin",  # conda 핵심 DLL 경로
        env / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        env / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        env / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
    ]

    added = []
    for d in candidates:
        if d.exists():
            os.add_dll_directory(str(d))
            added.append(str(d))

    # PATH에도 앞쪽에 추가(일부 라이브러리 대응)
    os.environ["PATH"] = ";".join(added) + ";" + os.environ.get("PATH", "")

    print("[DLL] added:", *added, sep="\n  - ")

add_cuda_dll_dirs()
# listen think speak Chain
from listen import SST_Wisper_Model
from thinking import model_thinking
from Speak_text import send_chatbox, set_typing
from Speak_voice import speak
SR = 16000
# ---- VAD(무음 감지) 파라미터 ----
FRAME_MS = 30  # 프레임 길이(ms) 20~30 권장
FRAME = int(SR * FRAME_MS / 1000)
START_SEC = 0.15  # 이만큼 연속으로 소리나면 "말 시작"
END_SEC = 1.5  # 이만큼 연속 무음이면 "말 종료"
MAX_SEC = 100  # 최대 녹음 길이(안전장치)

while True:
    listen_text = SST_Wisper_Model("Voicemeeter Out B1", FRAME_MS, FRAME, START_SEC, END_SEC, MAX_SEC, SR)
    print(listen_text)
    say_text = model_thinking("gemma3:1b", ask=listen_text)
    print("\n---\nRETURNED:", say_text)

    speak(say_text, out_device_keyword="Cable Input")
    set_typing(True)
    send_chatbox(f"AI_Bot:{say_text}", immediate=True, notify=False)
    set_typing(False)