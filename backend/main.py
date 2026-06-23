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
from pydantic import BaseModel, Field
import os
import google.generativeai as genai 
from googletrans import Translator, LANGUAGES
import pysubs2
from tqdm import tqdm
import whisperx
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS

class ProcessVideoRequest(BaseModel):
    s3_key: str
    num_clips: int = Field(default=3, ge=1, description="Number of clips to generate (minimum 1)")


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"])
    .add_local_dir("LR-ASD", "/LR-ASD", copy=True)
    .run_commands(["rm -rf /LR-ASD/.git"])) 

app = modal.App("clipzz", image=image)

volume = modal.Volume.from_name(
    "clipzz-model-cache", create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()



def get_video_fps(video_path):
    """
    Extract the frame rate (FPS) of a video file using ffprobe.

    Args:
        video_path (str): Path to the video file

    Returns:
        float: Frame rate in frames per second (e.g., 25.0, 29.97, 30.0)

    Example:
        >>> fps = get_video_fps("/path/to/video.mp4")
        >>> print(fps)  # 25.0
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "json", str(video_path)],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    fps_str = info["streams"][0]["r_frame_rate"]
    num, denom = map(int, fps_str.split("/"))
    return num / denom if denom != 0 else num



def create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, output_path, framerate=25):
    """
    Convert horizontal video to vertical format (1080x1920) with active speaker tracking.

    This function implements smart cropping that follows the active speaker or creates
    a blurred letterbox background when no speaker is detected. Uses a 60-frame window
    (30 before + 30 after) to smooth speaker scores and prevent jittery camera movement.

    Algorithm:
        1. Load face tracks and speaking scores from LR-ASD
        2. For each frame:
           - Calculate average speaking score over 60-frame window
           - Identify face with highest average score (active speaker)
           - CROP MODE: If speaker detected (score > 0), crop 1080px centered on face
           - RESIZE MODE: Otherwise, resize with blurred background letterboxing
        3. Merge processed video with original audio

    Args:
        tracks (list): Face track data from LR-ASD (pickle file)
            Each track contains:
                - track["track"]["frame"]: Frame indices where face appears
                - track["proc_track"]["x"]: X coordinates of face center
                - track["proc_track"]["y"]: Y coordinates of face center
                - track["proc_track"]["s"]: Face bounding box size
        scores (list): Per-frame speaking scores from LR-ASD (same length as tracks)
        pyframes_path (str): Directory containing extracted frames (*.jpg)
        pyavi_path (str): Directory to save output video
        audio_path (str): Path to audio file to merge with video
        output_path (str): Path to save final video with audio
        framerate (int, optional): Target frame rate. Default: 25 FPS

    Returns:
        None: Saves processed video to output_path

    Technical Details:
        - Target resolution: 1080x1920 (9:16 aspect ratio for social media)
        - Scoring window: 60 frames (~2.4s at 25fps) for stable tracking
        - GPU acceleration: Uses ffmpegcv.VideoWriterNV for fast encoding
        - Blur kernel: 121x121 Gaussian for background
        - Final encoding: H.264 with CRF 23, AAC audio at 128k

    Example:
        >>> create_vertical_video(
        ...     tracks=loaded_tracks,
        ...     scores=loaded_scores,
        ...     pyframes_path="/tmp/clip/pyframes",
        ...     pyavi_path="/tmp/clip/pyavi",
        ...     audio_path="/tmp/clip/audio.wav",
        ...     output_path="/tmp/clip/output.mp4"
        ... )
    """
    target_width = 1080
    target_height = 1920

    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()

    # Initialize faces list: one entry per frame, each containing list of detected faces
    faces = [[] for _ in range(len(flist))]

    # Populate faces with track data and average scores over 60-frame window
    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            # Average score over 60-frame window (30 before, 30 after) for smooth tracking
            slice_start = max(fidx - 30, 0)
            slice_end = min(fidx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice)
                              if len(score_slice) > 0 else 0)

            faces[frame].append(
                {'track': tidx, 'score': avg_score, 's': track['proc_track']["s"][fidx], 'x': track['proc_track']["x"][fidx], 'y': track['proc_track']["y"][fidx]})

    temp_video_path = os.path.join(pyavi_path, "video_only.mp4")

    vout = None
    for fidx, fname in tqdm(enumerate(flist), total=len(flist), desc="Creating vertical video"):
        img = cv2.imread(fname)
        if img is None:
            continue

        current_faces = faces[fidx]

        max_score_face = max(
            current_faces, key=lambda face: face['score']) if current_faces else None

        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        if vout is None:
            vout = ffmpegcv.VideoWriterNV(
                file=temp_video_path,
                codec=None,
                fps=framerate,
                resize=(target_width, target_height)
            )

        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"

        if mode == "resize":
            # RESIZE MODE: No clear speaker detected
            # Strategy: Letterbox with blurred background for aesthetic appeal

            # Step 1: Resize original video to fit width (1080px)
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0] * scale)
            resized_image = cv2.resize(
                img, (target_width, resized_height), interpolation=cv2.INTER_AREA)

            # Step 2: Create blurred background to fill vertical space
            scale_for_bg = max(
                target_width / img.shape[1], target_height / img.shape[0])
            bg_width = int(img.shape[1] * scale_for_bg)
            bg_heigth = int(img.shape[0] * scale_for_bg)

            blurred_background = cv2.resize(img, (bg_width, bg_heigth))
            blurred_background = cv2.GaussianBlur(
                blurred_background, (121, 121), 0)  # Heavy blur (121x121 kernel)

            # Step 3: Crop background to target dimensions
            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_heigth - target_height) // 2
            blurred_background = blurred_background[crop_y:crop_y +
                                                    target_height, crop_x:crop_x + target_width]

            # Step 4: Overlay resized video in center (letterbox effect)
            center_y = (target_height - resized_height) // 2
            blurred_background[center_y:center_y +
                               resized_height, :] = resized_image

            vout.write(blurred_background)

        elif mode == "crop":
            # CROP MODE: Active speaker detected
            # Strategy: Crop 1080px width centered on speaker's face

            # Step 1: Scale video height to 1920px
            scale = target_height / img.shape[0]
            resized_image = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]

            # Step 2: Find horizontal crop position centered on speaker
            center_x = int(
                max_score_face["x"] * scale if max_score_face else frame_width // 2)
            # Ensure crop stays within frame boundaries
            top_x = max(min(center_x - target_width // 2,
                        frame_width - target_width), 0)

            # Step 3: Crop 1080px width from scaled video
            image_cropped = resized_image[0:target_height,
                                          top_x:top_x + target_width]

            vout.write(image_cropped)

    if vout:
        vout.release()

    ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                      f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                      f"{output_path}")
    subprocess.run(ffmpeg_command, shell=True, check=True, text=True)

def is_hindi(text):
    """Check if the text contains Hindi (Devanagari) characters."""
    hindi_range = range(0x0900, 0x097F + 1)
    return any(ord(char) in hindi_range for char in text)

# def translate_hindi_to_english(text):
#     """Translate Hindi text to English using googletrans."""
#     translator = Translator()
#     try:
#         translation = translator.translate(text, src='hi', dest='en')
#         return translation.text
#     except Exception as e:
#         print(f"Translation error: {e}")
#         return text  # Fallback to original text if translation fails

def transliterate_hindi_to_english(text):
    """
    Transliterate Hindi text from Devanagari script to Roman (ITRANS) script.

    This converts Hindi characters like "नमस्ते" to "namaste" for better subtitle
    readability on platforms that may not render Devanagari properly.

    Args:
        text (str): Hindi text in Devanagari script

    Returns:
        str: Transliterated text in Roman script (ITRANS format),
             or original text if transliteration fails

    Example:
        >>> transliterate_hindi_to_english("नमस्ते")
        'namaste'
        >>> transliterate_hindi_to_english("क्या")
        'kyA'
    """
    try:
        return transliterate(text, DEVANAGARI, ITRANS)
    except Exception as e:
        print(f"Transliteration error: {e}")
        return text  # Fallback to original text if transliteration fails


def create_subtitles_with_ffmpeg(transcript_segments: list, clip_start: float, clip_end: float, clip_video_path: str, output_path: str, max_words: int = 5):
    """
    Generate karaoke-style subtitles with word-by-word highlighting and burn into video.

    Creates ASS (Advanced SubStation Alpha) format subtitles with:
    - Word grouping (5 words per line for readability)
    - Karaoke effect: each word highlights yellow as it's spoken
    - Professional styling: Anton font, white text, shadow & outline
    - Bottom-center positioning

    Algorithm:
        1. Filter transcript segments within clip time range
        2. Group words into sentences (max_words per sentence)
        3. Calculate karaoke timing (\k tags) for each word
        4. Generate ASS subtitle file with styling
        5. Burn subtitles into video using ffmpeg

    Args:
        transcript_segments (list): Word-level transcript with start/end timestamps
            Format: [{"word": "hello", "start": 1.5, "end": 1.8}, ...]
        clip_start (float): Clip start time in seconds (relative to original video)
        clip_end (float): Clip end time in seconds
        clip_video_path (str): Path to input video (without subtitles)
        output_path (str): Path to save video with burned-in subtitles
        max_words (int, optional): Words per subtitle line. Default: 5

    Returns:
        None: Saves subtitled video to output_path

    Technical Details:
        - Format: ASS (Advanced SubStation Alpha)
        - Resolution: 1080x1920 (PlayResX/Y)
        - Font: Anton, 140pt
        - Colors (BGR format):
            - Primary: &HFFFFFF& (white)
            - Highlight: &H00FFFF& (yellow)
            - Shadow: &H000000& (black, 128 alpha)
        - Karaoke timing: \k<centiseconds> tags (e.g., \k50 = 0.5 seconds)
        - Alignment: 2 (bottom center)
        - Margins: 50px all sides
        - Encoding: H.264, CRF 23, fast preset

    Example:
        >>> segments = [
        ...     {"word": "Hello", "start": 1.0, "end": 1.5},
        ...     {"word": "world", "start": 1.6, "end": 2.0}
        ... ]
        >>> create_subtitles_with_ffmpeg(
        ...     segments, 0.0, 5.0, "input.mp4", "output.mp4", max_words=5
        ... )
        # Creates video with karaoke subtitles
    """
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") >= clip_start
                     and segment.get("start") < clip_end
                     ]

    sentences = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        word = segment.get("word", "").strip()
        word = word.lower().replace("|", "").replace(".", "").replace("\\n", " ").replace("\\t", " ").replace("\\", "")  # Clean escape sequences
        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if not word or seg_start is None or seg_end is None:
            print(f"Skipping invalid segment: {segment}")
            continue
            
        # print("about to check for hindi 001")
        # if is_hindi(word):
        #     print("checking for hindi")
        #     word = transliterate_hindi_to_english(word)
        #     print(f"Transliterated '{segment['word']}' to '{word}'")

        start_rel = max(0.0, seg_start - clip_start)
        end_rel = max(0.0, seg_end - clip_start)

        if end_rel <= 0:
            continue

        if start_rel >= end_rel:
            print(f"Skipping segment with invalid timing: start={start_rel}, end={end_rel}, word={word}")
            continue


        if not current_words:
            current_start = seg_start
            current_end = seg_end
            current_words = [(word, start_rel, end_rel, seg_start, seg_end)]
        elif len(current_words) < max_words:
            current_words.append((word, start_rel, end_rel, seg_start, seg_end))
            current_end = seg_end
        else:
            sentences.append((current_start, current_end, current_words))
            current_words = [(word, start_rel, end_rel, seg_start, seg_end)]
            current_start = seg_start
            current_end = seg_end

    if current_words:
        sentences.append((current_start, current_end, current_words))

    print(f"Grouped into {len(sentences)} sentences")

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
    new_style.primarycolor = pysubs2.Color(255, 255, 255)  # Explicitly white
    new_style.secondarycolor = pysubs2.Color(255, 255, 255)  # White for karaoke
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)
    new_style.alignment = 2
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 50
    new_style.spacing = 0.0

    subs.styles[style_name] = new_style

    # Replaced per-word subtitle events with a single event per sentence, using ASS karaoke tags (\k<duration>) to control highlighting 
    # Generate one subtitle event per sentence with karaoke timing
    for sentence_start, sentence_end, words in sentences:
        # Build karaoke text with spaces and explicit color reset
        karaoke_parts = []
        for i, (word, start_rel, end_rel, _, _) in enumerate(words):
            duration_cs = int((end_rel - start_rel) * 100)  # Seconds to centiseconds
            if duration_cs < 1:
                duration_cs = 1  # Minimum duration
            karaoke_text = f"{{\\k{duration_cs}\\b1\\c&H00FFFF&}}{word}{{\\b0\\c&HFFFFFF&}}"
            karaoke_parts.append(karaoke_text)
            if i < len(words) - 1:
                karaoke_parts.append(" ")  # Add space between words

        # Add explicit color reset at the start
        styled_text = f"{{\\c&HFFFFFF&}}{''.join(karaoke_parts)}"
        start_time = pysubs2.make_time(s=max(0.0, sentence_start - clip_start))
        end_time = pysubs2.make_time(s=sentence_end - clip_start)

        # print(f"Adding subtitle: '{styled_text}' from {start_time} to {end_time}")
        line = pysubs2.SSAEvent(start=start_time, end=end_time, text=styled_text, style=style_name)
        subs.events.append(line)

    subs.save(subtitle_path)

    ffmpeg_cmd = (f"ffmpeg -y -i {clip_video_path} -vf \"ass={subtitle_path}\" "
                  f"-c:v h264 -preset fast -crf 23 {output_path}")

    subprocess.run(ffmpeg_cmd, shell=True, check=True)


def process_clip(base_dir: str, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):
    """
    Process a single clip from start to finish: extract, detect speaker, format, subtitle, upload.

    This is the main orchestration function that coordinates all clip processing steps:
    1. Extract video segment based on timestamps
    2. Extract audio track
    3. Run LR-ASD active speaker detection
    4. Create vertical video with smart cropping
    5. Generate and burn karaoke subtitles
    6. Upload final clip to S3

    Args:
        base_dir (str): Working directory for temporary files
        original_video_path (str): Path to source video file
        s3_key (str): S3 key of original video (used to construct output key)
        start_time (float): Clip start timestamp in seconds
        end_time (float): Clip end timestamp in seconds
        clip_index (int): Clip number (0, 1, 2, ...) for naming
        transcript_segments (list): Word-level transcript data for subtitles
            Format: [{"word": "hello", "start": 1.5, "end": 1.8}, ...]

    Returns:
        None: Uploads processed clip to S3

    Directory Structure Created:
        base_dir/
        ├── clip_N/
        │   ├── clip_N_segment.mp4       # Extracted segment
        │   ├── pywork/                  # LR-ASD working directory
        │   │   ├── tracks.pckl          # Face tracking data
        │   │   └── scores.pckl          # Speaking scores
        │   ├── pyframes/                # Extracted frames with bounding boxes
        │   │   └── *.jpg
        │   └── pyavi/                   # Output directory
        │       ├── audio.wav            # Extracted audio
        │       ├── video_out_vertical.mp4   # Vertical video without subtitles
        │       └── video_with_subtitles.mp4 # Final output

    Processing Steps:
        1. Extract segment: ffmpeg cuts video from start_time to end_time
        2. Extract audio: PCM 16-bit, 16kHz mono for LR-ASD
        3. Run Columbia_test.py (LR-ASD inference) from /LR-ASD directory
        4. Load tracks.pckl and scores.pckl pickle files
        5. Create vertical video with active speaker tracking
        6. Generate ASS subtitles with karaoke timing
        7. Burn subtitles into video
        8. Upload to S3: {original_s3_dir}/clip_N.mp4

    Example:
        >>> process_clip(
        ...     base_dir="/tmp/run123",
        ...     original_video_path="/tmp/run123/input.mp4",
        ...     s3_key="user456/upload789/video.mp4",
        ...     start_time=125.0,
        ...     end_time=180.0,
        ...     clip_index=0,
        ...     transcript_segments=[...]
        ... )
        # Uploads to S3: user456/upload789/clip_0.mp4
    """
    clip_name = f"clip_{clip_index}"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Output S3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True,
                   check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

    columbia_command = (f"python Columbia_test.py --videoName {clip_name} "
                        f"--videoFolder {str(base_dir)} "
                        f"--pretrainModel weight/finetuning_TalkSet.model")

    columbia_start_time = time.time()
    subprocess.run(columbia_command, cwd="/LR-ASD", shell=True)
    columbia_end_time = time.time()
    print(
        f"Columbia script completed in {columbia_end_time - columbia_start_time:.2f} seconds")

    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or scores not found for clip")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    cvv_start_time = time.time()
    create_vertical_video(
        tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path
    )
    cvv_end_time = time.time()
    print(
        f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds")

    create_subtitles_with_ffmpeg(transcript_segments, start_time,
                                 end_time, vertical_mp4_path, subtitle_output_path, max_words=5)

    s3_client = boto3.client("s3")
    s3_client.upload_file(
        subtitle_output_path, "clipzz", output_s3_key)


@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("clipzz-secret")], volumes={mount_path: volume})
class Clipzz:
    @modal.enter()
    def load_model(self):
        print("Loading models")

        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en",
            device="cuda"
        )

        print("Transcription models loaded...")

        print("Creating gemini client...")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.gemini_model = genai.GenerativeModel("gemini-2.5-pro")
        print("Created gemini client...")

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        """
        Transcribe video audio to text with word-level timestamps using WhisperX.

        Pipeline:
            1. Extract audio from video (16kHz, mono, PCM)
            2. Load audio with whisperx
            3. Transcribe with WhisperX large-v2 model (auto language detection)
            4. Align words with timestamps using language-specific alignment model
            5. For Hindi: transliterate Devanagari → Roman script
            6. Handle missing timestamps with interpolation

        Args:
            base_dir (str): Working directory to save audio file
            video_path (str): Path to input video file

        Returns:
            str: JSON string containing word-level transcript segments
                Format: '[{"word": "hello", "start": 1.5, "end": 1.8}, ...]'

        Language Support:
            - Auto-detects language (80+ languages supported)
            - English and Hindi have been tested
            - Hindi words are transliterated to Roman script (ITRANS)

        Timestamp Interpolation:
            - If a word is missing start/end timestamps:
              * Use previous word's end time as start
              * Estimate 0.6s duration for end time
              * Ensure end > start (minimum 0.1s duration)

        Example:
            >>> transcript = clipzz.transcribe_video("/tmp/run", "/tmp/video.mp4")
            >>> segments = json.loads(transcript)
            >>> print(segments[0])
            {'word': 'Hello', 'start': 0.5, 'end': 0.9}
        """
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

        print("Starting transcription with WhisperX...")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(audio, batch_size=16)

        detected_language = result.get("language", "en")
        print(f"Detected language: {detected_language}")

        # Ensure alignment model matches detected language
        alignment_model, metadata = whisperx.load_align_model(
            language_code=detected_language, device="cuda"
        )

        
        result = whisperx.align(
            result["segments"],
            alignment_model,
            metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

        duration = time.time() - start_time
        print(f"Transcription and alignment took {duration} seconds")

        segments = []
        last_end = 0.0  # Track the last known end time for interpolation

        print("--------------------------------")
        print(result)
        print("--------------------------------")

        if "word_segments" in result:
            for i, segment in enumerate(result["word_segments"]):
                word = segment.get("word", "")
                if not word:
                    print(f"Warning: Empty word at index {i}, skipping word processing")
                    continue

                # Handle missing 'start' or 'end' keys
                start = segment.get("start")
                end = segment.get("end")

                if start is None or end is None:
                    if i > 0 and segments:
                        prev_segment = segments[-1]
                        prev_end = prev_segment["end"]
                        start = start if start is not None else prev_end
                        end = end if end is not None else start + 0.6  # Reduced duration for tighter subtitles
                    else:
                        start = start if start is not None else last_end
                        end = end if end is not None else start + 0.6
                    print(f"Warning: Inferred timing for word '{word}' at index {i}: start={start}, end={end}")

                # Ensure end is greater than start
                if end <= start:
                    end = start + 0.1  # Minimum duration to avoid overlap issues
                    print(f"Adjusted end time for word '{word}' to {end} to ensure positive duration")

                last_end = end  # Update last known end time

                # Transliterate if Hindi
                if detected_language == "hi" and is_hindi(word):
                    word = transliterate_hindi_to_english(word)

                segments.append({
                    "start": float(start),
                    "end": float(end),
                    "word": word
                })

        return json.dumps(segments, indent=2, ensure_ascii=False)
    
    def identify_moments(self, transcript: dict):
        """
        Use Google Gemini 2.5 Pro to identify engaging Q&A moments in transcript.

        Analyzes the full transcript to find clips that:
        - Are 30-60 seconds long (optimized for social media)
        - Contain complete questions and answers
        - Don't overlap with each other
        - Exclude greetings, farewells, and non-content moments

        Args:
            transcript (dict): Word-level transcript segments
                Format: [{"word": "hello", "start": 1.5, "end": 1.8}, ...]

        Returns:
            str: Gemini response text containing JSON array of clip boundaries
                Format: '[{"start": 125.0, "end": 178.0}, ...]'
                May include markdown code fences (```json...```) which are stripped later

        Prompt Strategy:
            - Focus on Q&A format conversations
            - Prefer longer clips (40-60s) over shorter ones
            - Align clip boundaries with sentence boundaries
            - Use only exact timestamps from input (no modification)
            - Return empty list [] if no valid clips found

        Example:
            >>> moments = clipzz.identify_moments(transcript_segments)
            >>> print(moments)
            '[{"start": 125.0, "end": 178.0}, {"start": 200.0, "end": 255.0}]'
        """
        print("inside identify moments func")
        response = self.gemini_model.generate_content("""
    This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

    Your task is to find and extract stories, or question and their corresponding answers from the transcript.
    Each clip should begin with the question and conclude with the answer.
    It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

    Please adhere to the following rules:
    - Ensure that clips do not overlap with one another.
    - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
    - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
    - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
    - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

    Avoid including:
    - Moments of greeting, thanking, or saying goodbye.
    - Non-question and answer interactions.

    If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

    The transcript is as follows:\n\n""" + str(transcript))
        print(f"Identified moments response: ${response.text}")
        return response.text

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        s3_key = request.s3_key
        num_clips = request.num_clips

        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect bearer token", headers={"WWW-Authenticate": "Bearer"})

        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # Download video file
        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        s3_client.download_file("clipzz", s3_key, str(video_path))

        # 1. Transcription
        transcript_segments_json = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript_segments_json)

        # 2. Identify moments for clips
        print("Identifying clip moments")
        identified_moments_raw = self.identify_moments(transcript_segments)

        cleaned_json_string = identified_moments_raw.strip()
        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()

        clip_moments = json.loads(cleaned_json_string)
        if not clip_moments or not isinstance(clip_moments, list):
            print("Error: Identified moments is not a list")
            clip_moments = []

        print(clip_moments)

        # Limit num_clips to available moments
        num_clips = min(num_clips, len(clip_moments))
        if num_clips == 0:
            print("No valid clips to process")
            return {"status": "success", "clips_processed": 0}

        # 3. Process clips
        for index, moment in enumerate(clip_moments[:num_clips]):
            if "start" in moment and "end" in moment:
                print("Processing clip" + str(index) + " from " +
                    str(moment["start"]) + " to " + str(moment["end"]))
                process_clip(base_dir, video_path, s3_key,
                            moment["start"], moment["end"], index, transcript_segments)

        if base_dir.exists():
            print(f"Cleaning up temp dir after {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)


@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper = Clipzz()

    url = ai_podcast_clipper.process_video.web_url

    payload = {
        "s3_key": "test1/modi.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123321"
    }

    response = requests.post(url, json=payload,
                             headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)