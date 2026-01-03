import ctypes
import site
from pathlib import Path

# site-packages 후보들을 전부 훑어서 dll을 찾는다 (가장 안전)
cands = [Path(p) for p in site.getsitepackages()] + [Path(site.getusersitepackages())]

dll = None
for base in cands:
    if not base.exists():
        continue
    hits = list(base.rglob("cudnn_ops64_9.dll"))
    if hits:
        dll = hits[0]
        break

print("FOUND dll:", dll)

if dll is None:
    raise FileNotFoundError("cudnn_ops64_9.dll not found in site-packages")

print("DLL exists:", dll.exists(), dll)

try:
    ctypes.WinDLL(str(dll))
    print("Loaded cudnn_ops64_9.dll OK")
except OSError as e:
    print("Failed to load cudnn_ops64_9.dll:", e)