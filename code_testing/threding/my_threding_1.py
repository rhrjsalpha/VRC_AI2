import threading
import time
import random

w1_ended = threading.Event()
w2_ended = threading.Event()

def worker_1(list_1):
    while True:
        if w2_ended.is_set():
            rand_int = random.randint(1, 10)
            print("w1",rand_int)
            list_1.append(rand_int)
            if rand_int == 10:
                w1_ended.set()
                w2_ended.clear()
                list_1 = []

def worker_2(list_2):
    while True:
        if w1_ended.is_set():
            rand_int = random.randint(1, 10)
            print("w2",rand_int)
            list_2.append(rand_int)
            if rand_int == 10:
                w2_ended.set()
                w1_ended.clear()
                list_2=[]

list_1 = []
list_2 = []
w2_ended.set()
w1 = threading.Thread(target=worker_1, args=(list_1,))
w2 = threading.Thread(target=worker_2, args=(list_2,))
w1.start()
w2.start()


