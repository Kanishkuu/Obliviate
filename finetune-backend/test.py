import json
import os
import time
import requests
import websocket
import threading

# ==== CONFIG ====
API_URL = "http://localhost:8000"
DATA_FILE = "sample.jsonl"

# ==== 1. Create a 1-line dataset ====
os.makedirs("uploads", exist_ok=True)
sample_data = {
    "prompt": "What is AI?",
    "completion": "Artificial intelligence is the simulation of human intelligence by machines."
}
with open(DATA_FILE, "w") as f:
    f.write(json.dumps(sample_data) + "\n")

print(f"✅ Created dataset: {DATA_FILE}")

# ==== 2. Upload dataset ====
files = {"file": open(DATA_FILE, "rb")}
resp = requests.post(f"{API_URL}/api/upload", files=files)
if resp.status_code != 200:
    raise SystemExit(f"❌ Upload failed: {resp.text}")
dataset_path = resp.json()["path"]
print(f"✅ Uploaded dataset to: {dataset_path}")

# ==== 3. Define tiny model and config ====
payload = {
    "dataset": dataset_path,
    "base_model": "microsoft/phi-1_5",   # smallest possible GPT-2 model
    "load_in_4bit": False,                 # no bitsandbytes, CPU safe
    "epochs": 1,
    "max_steps": 2,
    "batch_size": 1,
    "grad_accum_steps": 1,
    "learning_rate": 5e-5,
    "max_seq_len": 128,
    "logging_steps": 1
}

# ==== 4. Start fine-tuning job ====
resp = requests.post(f"{API_URL}/api/finetune", json=payload)
if resp.status_code != 200:
    raise SystemExit(f"❌ Job start failed: {resp.text}")

data = resp.json()
run_id = data["run_id"]
print(f"🚀 Job started: run_id={run_id}")

# ==== 5. Subscribe to WebSocket progress ====
def listen_ws():
    ws_url = f"ws://localhost:8000/ws/progress/{run_id}"
    ws = websocket.WebSocket()
    ws.connect(ws_url)
    print("📡 Listening for progress...\n")
    try:
        while True:
            msg = ws.recv()
            try:
                parsed = json.loads(msg)
                stage = parsed.get("stage")
                print(f"→ {stage}")
                if stage == "done":
                    print("\n🎉 Training complete!")
                    print(json.dumps(parsed["result"], indent=2))
                    break
            except Exception:
                print("Message:", msg)
    except Exception as e:
        print("❌ WebSocket closed:", e)

t = threading.Thread(target=listen_ws, daemon=True)
t.start()

# ==== 6. Wait for completion ====
while t.is_alive():
    time.sleep(1)
