import os
from fastapi import APIRouter, UploadFile, File
from ...settings import settings

router = APIRouter(prefix="/api", tags=["upload"])

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    os.makedirs(settings.uploads_dir, exist_ok=True)
    dest = os.path.join(settings.uploads_dir, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"path": dest}
