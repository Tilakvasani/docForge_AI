import os
from fastapi import Header, HTTPException

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

async def verify_api_key(x_api_key: str = Header(default=None)):
    if not x_api_key or x_api_key != GROQ_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")