from fastapi import FastAPI
from .summary import Summarizer
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = [
    
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def index():
    return {"message": "Hello, World!"}



class VideoRequest(BaseModel):
    video_path: str

@app.post("/summarizer")
def summarize_video(request: VideoRequest):
    summary = Summarizer().summarize(request.video_path, mode='clustering')
    return {"message": summary}
