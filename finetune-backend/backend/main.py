from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .settings import settings
from .config import *  # ensure dirs
# fix: import routers from the existing package backend.api.routers
from .api.routers.finetune import router as finetune_router
from .api.routers.upload import router as upload_router
from .api.routers.progress import router as ws_router

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

app.include_router(finetune_router)
app.include_router(upload_router)
app.include_router(ws_router)
