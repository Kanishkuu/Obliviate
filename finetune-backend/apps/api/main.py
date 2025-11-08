import os, json, asyncio
from fastapi import FastAPI, Depends, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from rq import Queue
from redis import Redis
from apps.settings import settings
from .schemas import FineTuneRequest, JobResponse
from .deps import get_queue, get_redis
from apps.worker.jobs import run_training_job
from apps.api.progress import CHANNEL_PREFIX

app = FastAPI(title="FineTune Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(",") if settings.allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/finetune", response_model=JobResponse)
def start_finetune(req: FineTuneRequest, q: Queue = Depends(get_queue)):
    job = q.enqueue(run_training_job, req.model_dump(), job_timeout="24h")
    run_id = req.run_id or job.id
    return JobResponse(run_id=run_id, job_id=job.id)

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    os.makedirs(settings.uploads_dir, exist_ok=True)
    dest = os.path.join(settings.uploads_dir, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"path": dest}

@app.websocket("/ws/progress/{run_id}")
async def ws_progress(ws: WebSocket, run_id: str, redis: Redis = Depends(get_redis)):
    await ws.accept()
    pubsub = redis.pubsub()
    pubsub.subscribe(CHANNEL_PREFIX + run_id)
    try:
        while True:
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg["type"] == "message":
                await ws.send_text(msg["data"].decode("utf-8"))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pubsub.close()
    except Exception as e:
        pubsub.close()
        await ws.close(code=1011)
