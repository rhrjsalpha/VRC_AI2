from __future__ import annotations

import re
import json
import html
from pathlib import Path
from typing import Tuple, List


# =========================
# 1) encoding auto-detect
# =========================
def hangul_score(s: str) -> int:
    score = len(re.findall(r"[가-힣]", s))
    score -= s.count("�") * 5
    return score


def read_text_auto(path: Path) -> Tuple[str, str]:
    data = path.read_bytes()

    candidates = [
        "utf-8-sig",
        "utf-16",
        "cp949",
        "euc-kr",
        "utf-8",
        "latin-1",
    ]

    best_text = None
    best_enc = None
    best_score = -10**9

    for enc in candidates:
        try:
            text = data.decode(enc, errors="strict")
        except Exception:
            continue

        sc = hangul_score(text)
        if sc > best_score:
            best_score = sc
            best_text = text
            best_enc = enc

    if best_text is None:
        best_enc = "cp949"
        best_text = data.decode(best_enc, errors="replace")

    return best_text, best_enc


# =========================
# 2) short line handling
# =========================
def handle_short_lines(lines: List[str], min_len: int = 3, mode: str = "merge") -> List[str]:
    if mode == "drop":
        return [l for l in lines if len(l) >= min_len]

    if mode == "merge":
        merged = []
        for l in lines:
            if len(l) < min_len and merged:
                merged[-1] = merged[-1] + " " + l
            else:
                merged.append(l)
        return merged

    return lines


# =========================
# 3) SMI -> (start_ms, text)
# =========================
def parse_smi_to_segments(raw: str) -> List[Tuple[int, str]]:
    raw = html.unescape(raw)

    blocks = re.findall(
        r"<\s*sync[^>]*start\s*=\s*(\d+)[^>]*>(.*?)(?=<\s*sync\b|$)",
        raw,
        flags=re.I | re.S,
    )

    segs: List[Tuple[int, str]] = []
    for start_s, block in blocks:
        start = int(start_s)

        txt = block
        txt = re.sub(r"<\s*/?\s*p[^>]*>", "\n", txt, flags=re.I)
        txt = re.sub(r"<br\s*/?>", "\n", txt, flags=re.I)
        txt = re.sub(r"<[^>]+>", "", txt)
        txt = txt.replace("\u00a0", " ").replace("\u200b", "")
        txt = re.sub(r"[ \t]+", " ", txt)

        parts = []
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue
            s = re.sub(r"^\([^)]*\)\s*", "", s)   # (효과음)
            s = re.sub(r"^\[[^\]]*\]\s*", "", s)  # [SFX]
            s = s.strip()
            if s:
                parts.append(s)

        final = " ".join(parts).strip()
        if final:
            segs.append((start, final))

    return segs


# =========================
# 4) (start_ms, text) -> dialogues
# =========================
def merge_segments_into_dialogues(
    segs: List[Tuple[int, str]],
    *,
    time_threshold_ms: int = 800,
    min_len: int = 3,
    short_mode: str = "merge",
) -> List[str]:
    if not segs:
        return []

    end_re = re.compile(r"[.!?…]|(다|요|죠|네|나|까)\s*$")

    merged: List[str] = []
    buf = ""
    prev_t = None

    for t, txt in segs:
        if prev_t is None:
            buf = txt
        else:
            gap = t - prev_t
            if gap <= time_threshold_ms and not end_re.search(buf):
                buf = (buf + " " + txt).strip()
            else:
                merged.append(buf.strip())
                buf = txt

        prev_t = t

        if end_re.search(buf):
            merged.append(buf.strip())
            buf = ""
            prev_t = None

    if buf.strip():
        merged.append(buf.strip())

    merged = handle_short_lines(merged, min_len=min_len, mode=short_mode)
    return merged


def smi_to_dialogue_lines(
    path: Path,
    *,
    time_threshold_ms: int = 800,
    short_min_len: int = 3,
    short_mode: str = "merge",
) -> List[str]:
    raw, _enc = read_text_auto(path)
    segs = parse_smi_to_segments(raw)
    return merge_segments_into_dialogues(
        segs,
        time_threshold_ms=time_threshold_ms,
        min_len=short_min_len,
        short_mode=short_mode,
    )


# =========================
# 5) 1단계용 JSONL: {"text": "..."} 만 저장
#    - "대사 N줄을 하나의 샘플"로 묶어 저장(컨텍스트 학습에 유리)
# =========================
def write_pretrain_jsonl_from_dialogues(
    *,
    dialogues: List[str],
    out_jsonl: Path,
    chunk_size: int = 16,      # 한 샘플에 몇 줄 넣을지
    overlap: int = 4,          # 다음 샘플과 몇 줄 겹칠지(연결감)
    max_chars: int = 2000,     # 너무 길면 잘라내기(안전장치)
) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    n = 0
    step = chunk_size - overlap

    with out_jsonl.open("a", encoding="utf-8") as f:
        for i in range(0, len(dialogues), step):
            chunk = dialogues[i:i + chunk_size]
            if len(chunk) < 2:
                continue

            text = "\n".join(chunk).strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars].rstrip()

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1

    return n


# =========================
# 6) main: Data/ 아래 모든 .smi -> pretrain.jsonl
# =========================
def build_pretrain_dataset(
    data_root: str = "Data",
    out_jsonl: str = "pretrain.jsonl",
    *,
    time_threshold_ms: int = 800,
    short_mode: str = "merge",
    short_min_len: int = 3,
    chunk_size: int = 16,
    overlap: int = 4,
    max_chars: int = 2000,
):
    data_root_p = Path(data_root)
    out_jsonl_p = Path(out_jsonl)

    if out_jsonl_p.exists():
        out_jsonl_p.unlink()

    smi_files = sorted(data_root_p.rglob("*.smi"))
    print(f"[SCAN] found {len(smi_files)} smi files under: {data_root_p.resolve()}")

    total_samples = 0
    total_dialogues = 0

    for idx, smi in enumerate(smi_files, 1):
        raw, enc = read_text_auto(smi)
        segs = parse_smi_to_segments(raw)
        dialogues = merge_segments_into_dialogues(
            segs,
            time_threshold_ms=time_threshold_ms,
            min_len=short_min_len,
            short_mode=short_mode,
        )

        total_dialogues += len(dialogues)

        samples = write_pretrain_jsonl_from_dialogues(
            dialogues=dialogues,
            out_jsonl=out_jsonl_p,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chars=max_chars,
        )

        total_samples += samples
        print(
            f"[{idx}/{len(smi_files)}] {smi.name} | enc={enc:10s} | dialogues={len(dialogues):5d} | samples={samples:5d}"
        )

    print(f"[DONE] dialogues={total_dialogues} | samples={total_samples} -> {out_jsonl_p.resolve()}")


if __name__ == "__main__":
    build_pretrain_dataset(
        data_root="E:\VRC_AI2\Data",
        out_jsonl="pretrain.jsonl",
        time_threshold_ms=800,   # 500~1200 조절
        short_mode="merge",
        short_min_len=3,
        chunk_size=16,           # 8~32 추천
        overlap=4,               # 0~(chunk_size-1)
        max_chars=2000,          # 모델 컨텍스트 짧으면 1000~1500
    )
