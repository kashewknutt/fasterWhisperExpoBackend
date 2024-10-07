from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import os
import torch
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

if not os.path.exists("temp"):
    os.makedirs("temp")

os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.getcwd(), "model_cache")
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP errors

try:
    model = WhisperModel("large-v2", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    exit(1)  # Exit if model loading fails

@app.route('/transcribe/', methods=['POST'])
def transcribe_audio():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        audio_path = os.path.join("temp", file.filename)
        file.save(audio_path)

        segments, info = model.transcribe(audio_path)
        transcription = " ".join(segment.text for segment in segments)

        return jsonify({"transcription": transcription})
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return jsonify({"error": "Transcription failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8980, debug=True)
