from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.core.config import settings
from backend.core.logger import logger

app = FastAPI(
    title="AI Doc Generator",
    description="Generate industry-ready documents using LangChain + Groq and store in Notion",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
def root():
    logger.info("Root endpoint hit")
    return {"message": "AI Doc Generator API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}
