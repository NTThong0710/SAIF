import json
from datetime import datetime

def log_prompt(prompt, result, safe, response):
    log = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "result": result,
        "is_safe": safe,
        "response": response
    }
    with open("prompt_logs.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
