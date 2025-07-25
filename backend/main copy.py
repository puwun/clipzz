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
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
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
    .add_local_dir("LR-ASD", "/LR-ASD", copy=True)
    .run_commands(["rm -rf /LR-ASD/.git"]))  # Remove the .git directory after copying


app = modal.App("clipzz", image=image)

volume = modal.Volume.from_name(
    "clipzz-model-cache", create_if_missing=True
) 

# when whisperX downloads its model file for the first time, it will store it in this volume
mount_path = "/root/.cache/torch" 

auth_scheme = HTTPBearer()




def create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, output_path, frame_rate=None):
    target_width = 1080
    target_height = 1920

    f_list = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    f_list.sort()

    faces = [[] for _ in range(len(f_list))]

    for t_idx, track in enumerate(tracks):
        score_array = scores[t_idx]
        for f_idx, frame in enumerate(track["track"]["frame"].tolist()):
            slice_start = max(f_idx - 30, 0)
            slice_end = min(f_idx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice)
                              if len(score_slice) > 0 else 0)
            # Now we have 
            faces[frame].append({
                "track": t_idx, "score" : avg_score, "s" : track['proc_track']["s"][f_idx], "x" : track['proc_track']["x"][f_idx], "y" : track['proc_track']["y"][f_idx]})



    temp_vid_path = os.path.join(pyavi_path, "video_only.mp4")
    print(f"Creating vertical video at {temp_vid_path} with target size {target_width}x{target_height}")
    v_out = None

    # tqdm is used to show progress bar
    for f_idx, f_name in tqdm(enumerate(f_list), total=len(f_list), desc="Creating vertical video"):    
        frame = cv2.imread(f_name)
        if frame is None:
            print(f"Warning: Frame {f_name} could not be read. Skipping.")
            continue
        
        current_faces = faces[f_idx]
       
        # Ensure current_faces contains valid dictionaries with "score" key
        if current_faces and all(isinstance(face, dict) and "score" in face for face in current_faces):
            max_score_face = max(current_faces, key=lambda face: face["score"])
            if max_score_face["score"] < 0:
                max_score_face = None
                print(f"Warning: No faces with score > 0 in frame {f_idx}. Skipping.")
        else:
            max_score_face = None
            print(f"Warning: Invalid or empty faces in frame {f_idx}. Skipping.")
        
        #  ffmpegcv its just like ffmpeg, but it lets us GPU accelerate the video writing of the actual video (processing becomes a bit faster)
        if frame_rate is None:
            frame_rate = get_video_fps(f_name)
        
        if v_out is None:
            v_out = ffmpegcv.VideoWriterNV(
                file=temp_vid_path,
                codec=None,
                fps=frame_rate,
                resize=(target_width, target_height),
            )
        # if frame contains speaking face then we render it in crop mode, if not present then we render it in resize mode
        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"
        
        if mode == "resize":
            scale = target_width / frame.shape[1]
            resized_height = int(frame.shape[0] * scale)
            resized_frame = cv2.resize(frame, (target_width, resized_height), interpolation=cv2.INTER_AREA)

            scale_for_bg = max(
                target_width / frame.shape[1], target_height / frame.shape[0])
            bg_width = int(frame.shape[1] * scale_for_bg)
            bg_height = int(frame.shape[0] * scale_for_bg)

            blurred_bg = cv2.resize(frame, (bg_width, bg_height))
            blurred_bg = cv2.GaussianBlur(blurred_bg, (121, 121), 0)  #  Apply Gaussian blur to the background with a kernel size of (121, 121) 

            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_height - target_height) // 2

            blurred_bg = blurred_bg[crop_y:crop_y + target_height, crop_x:crop_x + target_width] 

            center_y = (target_height - resized_height) // 2
            blurred_bg[center_y:center_y + resized_height, :] = resized_frame
        
            v_out.write(blurred_bg)
            # print(f"Writing frame {f_idx} in resize mode with blurred background.")

        elif mode == "crop":
            scale = target_width / frame.shape[0]
            resized_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_frame.shape[1]

            center_x = int(max_score_face["x"] * scale if max_score_face else frame_width // 2)
            top_x = max(min(center_x - target_width // 2, frame_width - target_width), 0)

            frame_cropped = resized_frame[0: target_height, top_x:top_x + target_width]     # we want full height, but only a part of the width so crop from left and right
            frame_cropped = cv2.resize(frame_cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # v_out.write(frame_cropped)
            print(f"Writing frame {f_idx} in crop mode with face at x={max_score_face['x']}.")
    
    print("-----------------------------------")
    print(f"Total frames written: {len(f_list)}")
    print(f"Expected duration: {len(f_list) / frame_rate} seconds")
    print("Finished writing frames to video.")
    print("-----------------------------------")
    if v_out:
        v_out.release()

    print(f"Vertical video created at {temp_vid_path}")

    # print("before audio add cmd")
    #  Now we need to add audio to the video
    ffmpeg_cmd = (f"ffmpeg -y -i {temp_vid_path} -i {audio_path} "
                  f"-c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k "
                  f"-shortest {output_path}")
    subprocess.run(ffmpeg_cmd, shell=True, check=True, text=True)
    # print("after audio add cmd")


def create_subtitles(transcript_segments: list,clip_start: float, clip_end: float, clip_vid_path: str, output_path: str, max_words: int = 5):
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    clip_segments = [segment for segment in transcript_segments 
                     if segment["start"] is not None 
                     and segment["end"] is not None
                     and segment['end'] > clip_start
                     and segment['start'] < clip_end
                     ]
    subtitles = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        word = segment.get("word", "").strip()
        word = word.lower().replace("|", "").replace("\\n", " ").replace("\\t", " ").replace("\\", "")  # Clean escape sequences
        seg_start = segment.get("start", None)
        seg_end = segment.get("end", None)

        if not word or seg_start is None or seg_end is  None:
            continue
        
        start_relative = max(0.0, seg_start - clip_start)
        end_relative = max(0.0, seg_end - clip_start)

        if end_relative <= 0:
            continue
        
        if not current_words:
            current_start = start_relative
            current_end = end_relative
            current_words = [word]
        elif len(current_words) >= max_words:
            subtitles.append(( current_start, current_end, " ".join(current_words)))
            current_words = [word]
            current_start = start_relative
            current_end = end_relative
        else:
            current_words.append(word)
            current_end = end_relative

    if current_words:
        subtitles.append(( current_start, current_end, " ".join(current_words)))

    # Write subtitles to file
    subs = pysubs2.SSAFile()
    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    style_name = "Default"

    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Anton"
    new_style.fontsize = 140
    new_style.primarycolor = pysubs2.Color(255, 255, 255)  # White
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadow_color = pysubs2.Color(0, 0, 0, 128)  # Black shadow
    new_style.alignment = 2  # Centered vertically and horizontally
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 50
    new_style.spacing = 0.0

    subs.styles[style_name] = new_style

    for i, (start, end, text) in enumerate(subtitles):
        start_time = pysubs2.make_time(s=start)
        end_time = pysubs2.make_time(s=end)
        line = pysubs2.SSAEvent(start=start_time, end=end_time, style=style_name, text=text)
        subs.events.append(line)
    
    subs.save(subtitle_path)
    print(f"Subtitles created at {subtitle_path}")

    # print("before subtitle ffmpeg cmd exec")
    ffmpeg_cmd = (f"ffmpeg -y -i {clip_vid_path} -vf \"ass={subtitle_path}\" "
                    f"-c:v h264 -preset fast -crf 23 {output_path}")
    subprocess.run(ffmpeg_cmd, shell=True, check=True)
    # print("after subtitle ffmpeg cmd exec")

def process_clip(base_dir: pathlib.Path, video_path: pathlib.Path, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):   
# def process_clip(base_dir: str, video_path: str, s3_key: str, start: float, end: float, index: int, transcript_segments: list):   
    clip_name = f"clip_{clip_index}"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print("Output S3 Key:", output_s3_key)

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    #  Columbia script that creates a couple of folders for us, one of them is pyavi
    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)
    print("Clip directory created:", clip_dir)

    #  cutting the segment from the actual video
    duration = end_time - start_time
    print(f"Processing clip {clip_index} from {start_time} to {end_time} seconds, duration: {duration} seconds")
    # print("before ffmpeg cmd exec")
    cut_cmd = (f"ffmpeg -i {video_path} -ss {start_time} -t {duration} "
               f"{clip_segment_path}")
    subprocess.run(cut_cmd, shell=True, check=True, capture_output=True, text=True)
    # print("after ffmpeg cmd exec")

    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")
    
    columbia_cmd = (f"python Columbia_test.py --videoName {clip_name} "
                    f"--videoFolder {str(base_dir)} "
                    f"--pretrainModel weight/finetuning_TalkSet.model") 

    columbia_start_time = time.time()
    subprocess.run(columbia_cmd, cwd="/LR-ASD", shell=True)
    columbia_end_time = time.time()
    print(f"Columbia processing time for clip {clip_index}, {clip_name}: {columbia_end_time - columbia_start_time:.2f} seconds")

    os.listdir(clip_dir) #  this is just to see what files were created by Columbia_test script

    #  the output of Columbia_test script are some temporary files, one of which is tracks and other is scores that are stored as pkl files, which are just serialized python objects
    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"

    print(f"Tracks path: {tracks_path}, Scores path: {scores_path}")
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError(
            f"Columbia_test script did not create expected files: {tracks_path} or {scores_path}")      
    # print("done till here 01")
    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    print("Tracks and scores loaded from Columbia_test script output.")

    cvv_start_time = time.time()
    create_vertical_video(
        tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path)
    cvv_end_time = time.time()
    print(f"create_vertical_video processing time for clip {clip_index}, {clip_name}: {cvv_end_time - cvv_start_time:.2f} seconds")

    create_subtitles(transcript_segments, start_time, end_time, vertical_mp4_path, subtitle_output_path, max_words=5)


    s3_client = boto3.client("s3")
    s3_client.upload_file(
        subtitle_output_path, "clipzz", output_s3_key)


@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("clipzz-secret")],  volumes={mount_path: volume})
class Clipzz:
    @modal.enter()
    def load_model(self):
        print("Loading model...")

        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16") # later try with large-v3 as it was suggested by auto pilot to be better for long videos

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en", device="cuda")

        print("Creating Gemini client...")
        self.gemini_client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        print("Gemini client created...")


        print("Transcription Models loaded...")

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        print("BEFORE ffmpeg cmd exec")
        subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

        print("after ffmpeg cmd exec")


        # if result.returncode != 0:
        #     print(f"FFmpeg error: {result.stderr}")
        #     raise subprocess.CalledProcessError(result.returncode, extract_cmd, result.stdout, result.stderr)


        print("Started transcribing video mit WhisperX...")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(
            audio, batch_size=16) # language="en", word_timestamps=True) more the batch size, more poweer is needed for GPU

        detected_language = result.get("language", "en")  # Detect the language dynamically
        print(f"Detected language: {detected_language}")

        result = whisperx.align(
            result["segments"], 
            self.alignment_model, 
            self.metadata, 
            audio, 
            device="cuda",
            return_char_alignments=False)

        duration = time.time() - start_time

        print("Transcription and alignment completed in", duration, "seconds")

        # print(json.dumps(result, indent=2))

        segments = []

        if "word_segments" in result:
            for segment in result["word_segments"]:
                word = segment["word"]
                if detected_language == "hi":  # Transliterate Hindi words
                    word = transliterate(word, DEVANAGARI, ITRANS)
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "word": word
                })

        return json.dumps(segments, indent=2, ensure_ascii=False)
        # return json.dumps(segments)


    def identify_moment_for_clips(self, transcript : dict):
        # print("inside identify_moment_for_clips func")
        response = self.gemini_client.models.generate_content(model="gemini-2.5-pro", 
        contents="""
This is a podcast video transcript consisting of words, along with each word's start and end time. I am looking to create clips between a minimum of 40 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

Your task is to find and extract stories, or questions and their corresponding answers from the transcript.
Each clip should begin with the question and conclude with the answer.
It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

Please adhere to the following rules:
- Ensure that clips do not overlap with one another.
- Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
- Only use the start and end timestamps provided in the input. Modifying timestamps is not allowed.
- Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the Python json.loads function.
- Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

Avoid including:
- Moments of greeting, thanking, or saying goodbye.
- Non-question and answer interactions.

If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

The transcript is as follows:\n\n
""" + str(transcript) )

        print("Gemini response:", response.text)
        return response.text



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
        print(os.listdir(base_dir))

        # Transcription
        transcript_segments_json = self.transcribe_video(base_dir, video_path)

        transcript_segments = json.loads(transcript_segments_json)
        # print("Transcript segments:", transcript_segments)

        # Identify moment for clips
        print("Identifying moments for clips...")
        identified_moments_raw = self.identify_moment_for_clips(transcript_segments)

        cleaned_json_string = identified_moments_raw.strip()
        print("cleaned json 01")

        # Remove invalid entries like "RESULT: None"
        if "RESULT: None" in cleaned_json_string:
            cleaned_json_string = cleaned_json_string.replace("RESULT: None", "").strip()


        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
            print("cleaned json 02")
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()
            print("cleaned json 03")

        print("------------------------", cleaned_json_string, "------------------------")
       # Sanitize the JSON string
        cleaned_json_string = cleaned_json_string.replace('\\"', '"')  # Unescape any escaped quotes
        # cleaned_json_string = cleaned_json_string.replace('"', '\\"')  # Escape all quotes inside strings
        # cleaned_json_string = cleaned_json_string.replace("\\\\", "\\")  # Ensure backslashes are properly escaped

        # Parse the JSON
        try:
            identified_moments = json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Invalid JSON string: {cleaned_json_string}")
            identified_moments = []
        # print("Identified moments 01")
        
        if not identified_moments or not isinstance(identified_moments, list):
            print("Error: Identified moments is not a list. Returning empty list.")
            identified_moments = []
        
        print("Identified moments:", identified_moments)
        print("Starting to process identified moments...")
        # Process each identified moment
        for index, moment in enumerate(identified_moments[:7]):
            if "start" in moment and "end" in moment:
                print(f"Processing moment {index}: {moment['start']} - {moment['end']}")
                process_clip(base_dir, video_path, s3_key, moment['start'], moment['end'], index, transcript_segments)      # transcript_segments passed to process_clip for subtitle generation
            
        print("All identified moments processed.")
        if base_dir.exists():
            print(f"Temporary directory cleaned up: {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)


@app.local_entrypoint()
def main():
    import requests

    clipzz = Clipzz()

    url = clipzz.process_video.get_web_url()

    payload = {
        "s3_key": "test1/wealth_5min.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123321"
    }


    response = requests.post(url, json=payload, headers=headers)

    response.raise_for_status()

    result = response.json()

    print("RESULT:", result)  