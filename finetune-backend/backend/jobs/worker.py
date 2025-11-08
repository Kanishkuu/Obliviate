import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from rq import SimpleWorker, Queue, Connection
from redis import Redis, exceptions
from urllib.parse import urlparse
from ..settings import settings

listen = [settings.rq_queue]

def make_redis_client(url: str) -> Redis:
    r = Redis.from_url(url)
    try:
        r.ping()
        return r
    except exceptions.ConnectionError:
        # If configured hostname is the Docker service name 'redis' and it's not resolvable locally,
        # try a local fallback to 127.0.0.1 (useful when running outside Docker).
        parsed = urlparse(url)
        if parsed.hostname == "redis":
            fallback = url.replace("redis://redis", "redis://127.0.0.1")
            r2 = Redis.from_url(fallback)
            try:
                r2.ping()
                return r2
            except exceptions.ConnectionError:
                pass
        # re-raise the original exception if no fallback worked
        raise

if __name__ == "__main__":
    redis = make_redis_client(settings.redis_url)
    with Connection(redis):
        worker = SimpleWorker(list(map(Queue, listen)))
        worker.work(with_scheduler=True)
