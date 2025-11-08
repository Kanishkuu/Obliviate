import json
from redis import Redis, exceptions
from urllib.parse import urlparse
from ..settings import settings

CHANNEL_PREFIX = "progress:"

def _make_redis_client(url: str) -> Redis:
    r = Redis.from_url(url)
    try:
        r.ping()
        return r
    except exceptions.ConnectionError:
        parsed = urlparse(url)
        if parsed.hostname == "redis":
            fallback = url.replace("redis://redis", "redis://127.0.0.1")
            r2 = Redis.from_url(fallback)
            try:
                r2.ping()
                return r2
            except exceptions.ConnectionError:
                pass
        raise

def get_redis() -> Redis:
    return _make_redis_client(settings.redis_url)

def channel_name(run_id: str) -> str:
    return CHANNEL_PREFIX + run_id

def publish(run_id: str, payload: dict):
    r = get_redis()
    r.publish(channel_name(run_id), json.dumps(payload))
