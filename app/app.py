from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.routers.uce import router as test_router


API = FastAPI(title="Helical App", description="Test application with FastAPI")
API.include_router(test_router, prefix="/uce")

@API.get("/")
async def homepage() -> RedirectResponse:
    return RedirectResponse(url="/docs")