import uuid
from fastapi import APIRouter, Depends
from rq import Queue
from ..schemas import FineTuneRequest, JobResponse
from ..dependencies import get_queue
from ...services.training_service import enqueue_training

router = APIRouter(prefix="/api", tags=["finetune"])

@router.post("/finetune", response_model=JobResponse)
def start_finetune(req: FineTuneRequest, q: Queue = Depends(get_queue)):
    run_id = req.run_id or str(uuid.uuid4())
    job = enqueue_training(q, run_id, req)
    return JobResponse(run_id=run_id, job_id=job.id)
