import os
import uuid
import glob
import threading
import time
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from openai import OpenAI
from werkzeug.utils import secure_filename

from pydub import AudioSegment

from config import Config
from agents.agent_decision import process_query

config = Config()

limiter = Limiter(key_func=get_remote_address, default_limits=[f"{config.api.rate_limit}/minute"])
app = FastAPI(title="Multi-Agent Medical Chatbot", version="2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

UPLOAD_FOLDER = "uploads/backend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
BRAIN_TUMOR_OUTPUT = "uploads/brain_tumor_output"
SPEECH_DIR = "uploads/speech"

for d in [UPLOAD_FOLDER, "uploads/frontend", SKIN_LESION_OUTPUT, BRAIN_TUMOR_OUTPUT, SPEECH_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _cleanup_audio():
    while True:
        try:
            for f in glob.glob(f"{SPEECH_DIR}/*.mp3"):
                os.remove(f)
        except Exception as e:
            print(f"Audio cleanup error: {e}")
        time.sleep(300)


threading.Thread(target=_cleanup_audio, daemon=True).start()


class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []


class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat")
@limiter.limit(f"{config.api.rate_limit}/minute")
async def chat(request: Request, body: QueryRequest, response: Response, session_id: Optional[str] = Cookie(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
    try:
        data = process_query(body.query, session_id=session_id)
        response_text = data["messages"][-1].content
        response.set_cookie(key="session_id", value=session_id)
        result = {"status": "success", "response": response_text, "agent": data["agent_name"]}
        if "SKIN_LESION_AGENT" in str(data.get("agent_name", "")):
            seg_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(seg_path):
                result["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"
        if "BRAIN_TUMOR_AGENT" in str(data.get("agent_name", "")):
            seg_path = os.path.join(BRAIN_TUMOR_OUTPUT, "segmentation_plot.png")
            if os.path.exists(seg_path):
                result["result_image"] = "/uploads/brain_tumor_output/segmentation_plot.png"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
@limiter.limit(f"{config.api.rate_limit}/minute")
async def upload_image(
    request: Request,
    response: Response,
    image: UploadFile = File(...),
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None),
):
    if not allowed_file(image.filename):
        return JSONResponse(status_code=400, content={"status": "error", "agent": "System", "response": "Unsupported file type. Allowed: PNG, JPG, JPEG"})
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:
        return JSONResponse(status_code=413, content={"status": "error", "agent": "System", "response": f"File too large. Max: {config.api.max_image_upload_size}MB"})
    if not session_id:
        session_id = str(uuid.uuid4())
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    try:
        data = process_query({"text": text, "image": file_path}, session_id=session_id)
        response_text = data["messages"][-1].content
        response.set_cookie(key="session_id", value=session_id)
        result = {"status": "success", "response": response_text, "agent": data["agent_name"]}
        if "SKIN_LESION_AGENT" in str(data.get("agent_name", "")):
            seg_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(seg_path):
                result["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"
        if "BRAIN_TUMOR_AGENT" in str(data.get("agent_name", "")):
            seg_path = os.path.join(BRAIN_TUMOR_OUTPUT, "segmentation_plot.png")
            if os.path.exists(seg_path):
                result["result_image"] = "/uploads/brain_tumor_output/segmentation_plot.png"
        try:
            os.remove(file_path)
        except Exception:
            pass
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
@limiter.limit(f"{config.api.rate_limit}/minute")
async def validate_medical_output(
    request: Request,
    response: Response,
    validation_result: str = Form(...),
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None),
):
    if not session_id:
        session_id = str(uuid.uuid4())
    try:
        response.set_cookie(key="session_id", value=session_id)
        query = f"Validation result: {validation_result}"
        if comments:
            query += f" Comments: {comments}"
        data = process_query(query, session_id=session_id)
        content = data["messages"][-1].content
        if validation_result.lower() == "yes":
            return {"status": "validated", "message": "**Output confirmed by human validator:**", "response": content}
        return {"status": "rejected", "comments": comments, "message": "**Output requires further review:**", "response": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
@limiter.limit("5/minute")
async def transcribe_audio(request: Request, audio: UploadFile = File(...)):
    if not audio.filename:
        return JSONResponse(status_code=400, content={"error": "No audio file provided"})
    os.makedirs(SPEECH_DIR, exist_ok=True)
    temp_webm = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"
    audio_content = await audio.read()
    with open(temp_webm, "wb") as f:
        f.write(audio_content)
    if os.path.getsize(temp_webm) == 0:
        os.remove(temp_webm)
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})
    mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"
    try:
        AudioSegment.from_file(temp_webm).export(mp3_path, format="mp3")
        with open(mp3_path, "rb") as f:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1", file=f, language="en"
            )
        for p in [temp_webm, mp3_path]:
            try:
                os.remove(p)
            except Exception:
                pass
        if transcription.text:
            return {"transcript": transcription.text}
        return JSONResponse(status_code=500, content={"error": "Transcription returned empty"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate-speech")
@limiter.limit("10/minute")
async def generate_speech(request: Request, body: SpeechRequest):
    if not body.text:
        return JSONResponse(status_code=400, content={"error": "Text is required"})
    try:
        os.makedirs(SPEECH_DIR, exist_ok=True)
        audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        response = openai_client.audio.speech.create(
            model="tts-1", voice="alloy", input=body.text
        )
        response.stream_to_file(audio_path)
        return FileResponse(path=audio_path, media_type="audio/mpeg", filename="speech.mp3")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.exception_handler(413)
async def too_large(request: Request, exc):
    return JSONResponse(status_code=413, content={"status": "error", "agent": "System", "response": f"File too large. Max: {config.api.max_image_upload_size}MB"})


if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)
