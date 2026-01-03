import asyncio
import edge_tts
import sounddevice as sd
import soundfile as sf

def find_output_device(keyword: str):
    keyword = keyword.lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0 and keyword in dev["name"].lower():
            return i
    return None

async def tts_to_wav(text: str, wav_path: str, voice="ko-KR-SunHiNeural"):
    await edge_tts.Communicate(text, voice=voice).save(wav_path)

def play_wav(wav_path: str, out_device_keyword: str):
    dev_id = find_output_device(out_device_keyword)
    if dev_id is None:
        raise RuntimeError(f"Output device not found: {out_device_keyword}")

    data, sr = sf.read(wav_path, dtype="float32")
    sd.play(data, sr, device=dev_id)
    sd.wait()

def speak(text: str, out_device_keyword="INZONE", voice="ko-KR-SunHiNeural"):
    wav_path = "../tts.wav"
    asyncio.run(tts_to_wav(text, wav_path, voice=voice))
    play_wav(wav_path, out_device_keyword)

# 예시
# speak("출력 장치 지정 테스트", out_device_keyword="Cable Input") # "VoiceMeeter Input"
# speak(..., out_device_keyword="INZONE")

