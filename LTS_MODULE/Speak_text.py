from pythonosc.udp_client import SimpleUDPClient
import pyttsx3

VRC_IP = "127.0.0.1"
VRC_OSC_IN_PORT = 9000  # VRChat이 받는 기본 포트(대부분 9000)
client = SimpleUDPClient(VRC_IP, VRC_OSC_IN_PORT)

def send_chatbox(text: str, immediate: bool = True, notify: bool = False):
    # VRChat chatbox는 144자 제한이 있으니 잘라주는 게 안전
    text = text[:144]
    client.send_message("/chatbox/input", [text, bool(immediate), bool(notify)])

def set_typing(is_typing: bool):
    client.send_message("/chatbox/typing", bool(is_typing))

def say_tts(text: str):
    say_text = text
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)  # 말하기 속도
    engine.setProperty("volume", 1.0)

    engine.say("안녕하세요. 오프라인 TTS 테스트입니다.")
    engine.runAndWait()


if __name__ == "__main__":
    # 사용 예
    set_typing(True)
    send_chatbox("안녕하세요! OSC로 채팅박스 전송 테스트입니다.", immediate=True, notify=False)
    set_typing(False)