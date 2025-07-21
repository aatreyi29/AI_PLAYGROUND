from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Sample YouTube demo data
demos = [
    {"title": "Object Detection Demo", "url": "https://www.youtube.com/embed/YOUR_VIDEO_ID_1"},
    {"title": "Facial Recognition Demo", "url": "https://www.youtube.com/embed/YOUR_VIDEO_ID_2"},
    {"title": "Player Tracking Demo", "url": "https://www.youtube.com/embed/YOUR_VIDEO_ID_3"},
]

@router.get("/cv-demos", response_class=HTMLResponse)
async def get_cv_demos(request: Request):
    return templates.TemplateResponse("computer_vision_demos.html", {"request": request, "demos": demos})
