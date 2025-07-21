from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from app.api.router import api_router

app = FastAPI()

# Add this:
templates = Jinja2Templates(directory="app/templates")

# Include your routers
app.include_router(api_router)

# Optional: for serving static files if needed
app.mount("/static", StaticFiles(directory="app/static"), name="static")
