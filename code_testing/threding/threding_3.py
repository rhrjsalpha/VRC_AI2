import threading
import time

stop_event = threading.Event()

def worker():
    while not stop_event.is_set():
        print("돌아가는 중...")
        time.sleep(0.5)
    print("정지됨!")

t = threading.Thread(target=worker)
t.start()

time.sleep(2)
stop_event.set()  # 스레드에게 “멈춰” 신호
t.join()
print("메인 종료")
