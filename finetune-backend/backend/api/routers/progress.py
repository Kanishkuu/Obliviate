import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from redis import Redis
from ..dependencies import get_redis
from ...services.progress_service import channel_name

router = APIRouter(tags=["progress"])

@router.websocket("/ws/progress/{run_id}")
async def ws_progress(ws: WebSocket, run_id: str, redis: Redis = Depends(get_redis)):
    await ws.accept()
    pubsub = redis.pubsub()
    pubsub.subscribe(channel_name(run_id))
    try:
        while True:
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg and msg["type"] == "message":
                await ws.send_text(msg["data"].decode("utf-8"))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pubsub.close()
    except Exception:
        pubsub.close()
        await ws.close(code=1011)
