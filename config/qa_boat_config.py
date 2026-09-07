import json
import requests

QA_BOAT_URL = "http://192.168.1.81:11434/api/generate"
QA_BOAT_MODEL = "qwen3-coder:30b"

def generate_response_qa_boat(prompt):

    payload = {
        "model": QA_BOAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2
        }
    }

    response = requests.post(
        QA_BOAT_URL,
        json=payload,
        #timeout=120
    )

    response.raise_for_status()

    api_response = response.json()

    generated_testcases = json.loads(
        api_response["response"]
    )

    return json.dumps(generated_testcases, indent=2)