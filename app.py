from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import os
import torch

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Ensure the temp directory exists
if not os.path.exists("temp"):
    os.makedirs("temp")

# Set Hugging Face cache directory to use the current directory
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.getcwd(), "model_cache")
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')

# Load the Whisper model (choose the size based on available resources)
model = WhisperModel("large-v2", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the uploaded audio file in the current directory's temp folder
    audio_path = f"temp/{file.filename}"
    with open(audio_path, "wb") as audio_file:
        audio_file.write(await file.read())

    # Transcribe the audio
    segments, info = model.transcribe(audio_path)
    
    # Collect the transcription
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "

    return {"transcription": transcription}
