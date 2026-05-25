import requests
import torch
from transformers import pipeline

# Download sample audio file (if not already present)
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX04C6EN/Testing%20speech%20to%20text.mp3"
audio_file_path = "downloaded_audio.mp3"

response = requests.get(url)
if response.status_code == 200:
    with open(audio_file_path, "wb") as f:
        f.write(response.content)
    print("Sample audio downloaded successfully")
else:
    print("Failed to download sample audio")

# Transcribe using Whisper
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
)

prediction = pipe(audio_file_path, batch_size=8)["text"]
print("\nTranscription:\n", prediction)