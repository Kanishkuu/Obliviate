from redis import Redis, exceptions
from rq import Queue
from urllib.parse import urlparse
from ..settings import settings

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

def get_queue() -> Queue:
    # Create Queue with a Redis connection (the Queue constructor accepts a connection object)
    redis_client = get_redis()
    return Queue(settings.rq_queue, connection=redis_client)
