# 스레드에 인자 전달하기
import threading
import time

def worker(name, n):
    for i in range(n):
        print(f"[{name}] {i}")
        time.sleep(0.3)

t1 = threading.Thread(target=worker, args=("A", 5))
t2 = threading.Thread(target=worker, args=("B", 5))

t1.start()
t2.start()

t1.join()
t2.join()
print("둘 다 끝!")