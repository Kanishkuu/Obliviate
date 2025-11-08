import os
from rq import Worker, Queue, Connection
from redis import Redis
from apps.settings import settings

listen = [settings.rq_queue]

if __name__ == "__main__":
    redis = Redis.from_url(settings.redis_url)
    with Connection(redis):
        worker = Worker(list(map(Queue, listen)))
        worker.work(with_scheduler=True)
