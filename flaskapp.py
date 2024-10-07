from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import os
import torch

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Ensure the temp directory exists
if not os.path.exists("temp"):
    os.makedirs("temp")

# Set Hugging Face cache directory to use the current directory
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.getcwd(), "model_cache")
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')

# Load the Whisper model (choose the size based on available resources)
model = WhisperModel("large-v2", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

@app.route('/transcribe/', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded audio file in the current directory's temp folder
    audio_path = os.path.join("temp", file.filename)
    file.save(audio_path)

    # Transcribe the audio
    segments, info = model.transcribe(audio_path)
    
    # Collect the transcription
    transcription = " ".join(segment.text for segment in segments)

    return jsonify({"transcription": transcription})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8980, debug=True)