import glob
import json
import pathlib
import pickle
import shutil
import subprocess
import time
import uuid
import boto3
import cv2
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import ffmpegcv
import modal
import numpy as np
from pydantic import BaseModel
import os
from google import genai

import pysubs2
from tqdm import tqdm
import whisperx


class ProcessVideoRequest(BaseModel):
    s3_key: str


# Basic docker image that is used to create the environment that our endpoint runs within
image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"])
    .add_local_dir("LR-ASD", "/LR-ASD", copy=True))


app = modal.App("clipzz", image=image)

volume = modal.Volume.from_name(
    "clipzz-model-cache", create_if_missing=True
) 

# when whisperX downloads its model file for the first time, it will store it in this volume
mount_path = "/root/.cache/torch" 

auth_scheme = HTTPBearer()

@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("clipzz-secret")],  volumes={mount_path: volume})
class Clipzz:
    @modal.enter()
    def load_model(self):
        print("Loading model...")

        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16") # later try with large-v3 as it was suggested by auto pilot to be better for long videos

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en", device="cuda")

        print("Transcription Models loaded...")

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

        print("Started transcribing video mit WhisperX...")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(
            audio, batch_size=16) # language="en", word_timestamps=True) more the batch size, more poweer is needed for GPU
        result = whisperx.align(
            result["segments"], 
            self.alignment_model, 
            self.metadata, 
            audio, 
            device="cuda",
            return_char_alignments=False)

        duration = time.time() - start_time

        print("Transcription and alignment completed in", duration, "seconds")

        print(json.dumps(result, indent=2))


    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        s3_key = request.s3_key

        if token.credentials != os.environ.get("AUTH_TOKEN"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        print("Processing video:", s3_key)
        
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)
        
        print("Base directory created:", base_dir)
        
        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        s3_client.download_file("clipzz", s3_key, str(video_path))

        self.transcribe_video(base_dir, video_path)

        print(os.listdir(base_dir))

        pass


@app.local_entrypoint()
def main():
    import requests

    clipzz = Clipzz()

    url = clipzz.process_video.get_web_url()

    payload = {
        "s3_key": "test1/modi_5min.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }


    response = requests.post(url, json=payload, headers=headers)

    response.raise_for_status()

    result = response.json()

    print("RESULT:", result)  