import json

with open("hf_token.txt", encoding="utf8") as f:
    HF_TOKEN = f.read().strip()

with open("metrics.json", encoding="utf8") as f:
    METRICS = json.load(f)

with open("topics.json", encoding="utf8") as f:
    TOPICS = json.load(f)
