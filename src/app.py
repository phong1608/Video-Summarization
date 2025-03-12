from fastapi import FastAPI
from summary import Summarizer
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def index():
    return {"message": "Hello, World!"}



class VideoRequest(BaseModel):
    video_path: str

@app.post("/summarizer")
def summarize_video(request: VideoRequest):
    summary = Summarizer().summarize(request.video_path, mode='consine')
    return {"message": summary}
