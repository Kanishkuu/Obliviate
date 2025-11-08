from rq import Queue
from ..api.schemas import FineTuneRequest
from ..settings import settings
from ..jobs.tasks import run_training_job

def enqueue_training(q: Queue, run_id: str, req: FineTuneRequest):
    payload = req.model_dump()
    payload["run_id"] = run_id
    payload["artifacts_dir"] = settings.artifacts_dir
    job = q.enqueue(run_training_job, payload, job_timeout="24h")
    return job
