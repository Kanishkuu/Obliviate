from typing import Dict, Any
from redis import Redis
import json
from apps.settings import settings

CHANNEL_PREFIX = "progress:"

_redis = None
def get_redis():
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url)
    return _redis

def publish_progress(run_id: str, payload: Dict[str, Any]):
    r = get_redis()
    r.publish(CHANNEL_PREFIX + run_id, json.dumps(payload))
