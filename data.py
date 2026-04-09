import json, os

HF_TOKEN = None
if "hf_token.txt" in os.listdir():
    with open("hf_token.txt", encoding="utf8") as f:
        HF_TOKEN = f.read().strip()
else:
    print("! HF_TOKEN is None")

def float_keys(o: dict[str]):
    return {float(k) if k[0].isnumeric() else k: v for k, v in o.items()}

with open("metrics.json", encoding="utf8") as f:
    METRICS = json.load(f, object_hook=float_keys)

with open("topics.json", encoding="utf8") as f:
    TOPICS = json.load(f)

CURVE_TRANSLATIONS = {
    "Naive": "Простая оценка",
    "Weighted": "Взвешенная оценка"
}
