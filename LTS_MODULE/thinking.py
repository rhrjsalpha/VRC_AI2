import ollama

def model_thinking(model: str, ask: str) -> str:
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "답변은 2문장 이내로, 최대 40단어로만."},
            {"role": "user", "content": ask},
        ],
        stream=True,
    )

    parts = []
    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            #print(delta, end="", flush=True)  # 실시간 출력 유지
            parts.append(delta)
    return "".join(parts)

if __name__ == "__main__":
    text = model_thinking("gemma3:1b", ask="안녕?")
    print("\n---\nRETURNED:", text)