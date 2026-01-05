# 함수 하나를 백그라운드로 돌리기 (daemon thread)
import threading
import time

def worker():
    while True:
        print("작업중...")
        time.sleep(1)

t = threading.Thread(target=worker, daemon=True)
t.start()

print("메인 실행중...")
time.sleep(5)
print("끝")  # daemon=True라 프로그램 종료되면 스레드도 같이 종료