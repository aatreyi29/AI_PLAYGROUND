from fastapi import APIRouter
from .computer_vision import endpoints as cv_endpoints

api_router = APIRouter()
api_router.include_router(cv_endpoints.router, prefix="/cv", tags=["Computer Vision"])
