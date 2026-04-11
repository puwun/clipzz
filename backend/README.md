# <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f9ea/512.gif" width="32"> Clipzz Backend - Video Processing Pipeline

Python-based video processing backend running on Modal serverless GPU infrastructure.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4e6/512.gif" width="24"> Overview

The backend is responsible for the entire video processing pipeline:
1. Video transcription with WhisperX
2. AI-powered moment identification with Google Gemini
3. Active speaker detection with LR-ASD
4. Vertical video creation with smart cropping
5. Karaoke-style subtitle generation and burn-in

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f680/512.gif" width="24"> Technology Stack

- **Runtime:** Python 3.12
- **Framework:** FastAPI
- **Infrastructure:** Modal (Serverless GPU - L40S)
- **AI Models:**
  - WhisperX large-v2 (speech-to-text)
  - Google Gemini 2.5 Pro (content analysis)
  - LR-ASD (active speaker detection)
- **Video Processing:**
  - FFmpeg / ffmpegcv (encoding/decoding)
  - OpenCV (image manipulation)
  - PyTorch 2.4.1 + CUDA 12.4
- **Storage:** AWS S3 (boto3)

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4bb/512.gif" width="24"> Quick Start

### Prerequisites

- Python 3.12+
- Modal account ([sign up](https://modal.com))
- Google Gemini API key
- AWS S3 bucket and credentials

### Installation

```bash
# Clone and navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token new
```

### Initialize LR-ASD Submodule

```bash
# From root directory
git submodule update --init --recursive
```

### Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Configure environment variables (see [CONFIGURATION.md](../CONFIGURATION.md))

3. Create Modal secrets:
```bash
modal secret create clipzz-secrets \
  AWS_ACCESS_KEY_ID="your-key" \
  AWS_SECRET_ACCESS_KEY="your-secret" \
  GEMINI_API_KEY="your-gemini-key" \
  AUTH_TOKEN="your-auth-token"
```

### Running Locally

```bash
# Serve function locally (for development)
modal serve main.py
```

This will output a local endpoint URL you can test with:
```bash
curl -X POST "https://your-app--dev.modal.run/process_video" \
  -H "Authorization: Bearer your-auth-token" \
  -H "Content-Type: application/json" \
  -d '{"s3_key": "user-id/video-id/video.mp4", "num_clips": 3}'
```

### Deploying to Production

```bash
modal deploy main.py
```

Copy the production endpoint URL to your frontend's `.env`:
```bash
PROCESS_VIDEO_ENDPOINT="https://your-app--clipzz-video-processor-process-video.modal.run"
```

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f3d7/512.gif" width="24"> Architecture

### Modal App Structure

```python
app = modal.App("clipzz-video-processor")

# Persistent volume for model caching
volume = modal.Volume.from_name("clipzz-models", create_if_missing=True)

# Custom CUDA image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["ffmpeg", "git", "libsndfile1", ...])
    .pip_install(["torch==2.4.1+cu124", "whisperx", ...])
    .run_commands([...])
)

@app.function(
    gpu="L40S",           # NVIDIA L40S GPU
    timeout=900,          # 15 minutes
    image=image,
    volumes={"/root/.cache": volume}  # Cache models between runs
)
def process_video(request: ProcessVideoRequest, token: str):
    # Main processing logic
    ...
```

### Processing Pipeline

```
┌──────────────────────────────────────────────────────────┐
│ 1. Download Video from S3                               │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│ 2. Transcribe with WhisperX                             │
│    - Detect language (auto)                             │
│    - Transcribe audio → text                            │
│    - Align word-level timestamps                        │
│    - Transliterate Hindi (if detected)                  │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│ 3. Identify Moments with Gemini                         │
│    - Analyze full transcript                            │
│    - Extract engaging Q&A segments                      │
│    - Return 30-60s clip boundaries                      │
└───────────────────┬──────────────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │   For Each Clip      │
        └──────────┬───────────┘
                   │
    ┌──────────────▼──────────────────┐
    │ 4. Extract Video Segment        │
    │    - Cut video based on timing  │
    │    - Extract audio track        │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │ 5. Active Speaker Detection     │
    │    - Run LR-ASD model           │
    │    - Track faces across frames  │
    │    - Score speaking activity    │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │ 6. Create Vertical Video        │
    │    - Target: 1080x1920          │
    │    - Crop to active speaker     │
    │    - Or blur background         │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │ 7. Generate Subtitles           │
    │    - Group words (5 per line)   │
    │    - Calculate karaoke timing   │
    │    - Create ASS subtitle file   │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │ 8. Burn Subtitles               │
    │    - Merge subtitle + video     │
    │    - Final encode               │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │ 9. Upload to S3                 │
    │    - Save as clip_N.mp4         │
    └─────────────────────────────────┘
```

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4da/512.gif" width="24"> Core Functions

### `transcribe_video(base_dir, video_path)`
Transcribes video audio using WhisperX.

**Features:**
- Automatic language detection
- Word-level timestamp alignment
- Hindi transliteration (Devanagari → Roman)
- Handles missing timestamps with interpolation

**Returns:**
```python
{
    "segments": [
        {
            "start": 10.5,
            "end": 15.2,
            "text": "This is an example",
            "words": [
                {"word": "This", "start": 10.5, "end": 10.8},
                {"word": "is", "start": 10.9, "end": 11.0}
            ]
        }
    ]
}
```

### `identify_moments(transcript)`
Uses Google Gemini 2.5 Pro to identify engaging moments.

**Input:** Full transcript as text
**Output:** List of clip boundaries
```python
[
    {
        "clip_number": 1,
        "start_seconds": 125.0,
        "end_seconds": 178.0,
        "description": "Discussion about AI in education"
    }
]
```

**Prompt Strategy:**
- Focus on Q&A format conversations
- 30-60 second optimal length
- No overlapping clips
- Exclude greetings and intros

### `process_clip(...)`
Processes a single clip from start to finish.

**Steps:**
1. Extract video segment with ffmpeg
2. Extract audio track
3. Run LR-ASD active speaker detection
4. Create vertical video with smart cropping
5. Generate ASS subtitles with karaoke effect
6. Burn subtitles into video
7. Upload to S3

**Parameters:**
- `base_dir`: Working directory for temp files
- `original_video_path`: Path to source video
- `s3_key`: S3 key of original video
- `start_time`: Clip start timestamp (seconds)
- `end_time`: Clip end timestamp (seconds)
- `clip_index`: Clip number (1, 2, 3, ...)
- `transcript_segments`: Word-level transcript data

### `create_vertical_video(...)`
Converts horizontal video to vertical format with active speaker tracking.

**Target Resolution:** 1080x1920 (9:16 aspect ratio)

**Algorithm:**
1. Load face tracks and speaking scores from LR-ASD
2. For each frame:
   - Calculate average score over 60-frame window
   - Identify highest-scoring face (active speaker)
   - **Crop Mode:** If speaker detected, crop 1080px around face
   - **Resize Mode:** Otherwise, blur background and letterbox
3. GPU-accelerated encoding with ffmpegcv

**Window Size:** 60 frames (~2.4 seconds at 25fps) for stable tracking

### `create_subtitles_with_ffmpeg(...)`
Generates karaoke-style subtitles with word-level highlighting.

**Format:** ASS (Advanced SubStation Alpha)

**Features:**
- Groups words into 5-word sentences
- Yellow highlight on current word (`&H00FFFF&`)
- White text with shadow and outline
- Bottom-center positioning
- Anton font, 140pt

**Karaoke Timing:**
```ass
{\k50}Word1 {\k30}Word2 {\k40}Word3
```
Numbers are centiseconds (50 = 0.5 seconds).

**Burn-In:** Uses ffmpeg subtitles filter (hardware-accelerated if available)

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f527/512.gif" width="24"> API Reference

### POST /process_video

Processes a video and generates clips.

**Authentication:** Bearer token (required)
```
Authorization: Bearer your-auth-token
```

**Request Body:**
```json
{
  "s3_key": "user-id/upload-id/original.mp4",
  "num_clips": 3
}
```

**Response (Success):**
```json
{
  "status": "success",
  "clips_processed": 3
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Error description"
}
```

**Status Codes:**
- `200 OK`: Processing successful
- `401 Unauthorized`: Invalid or missing auth token
- `500 Internal Server Error`: Processing failed

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4c1/512.gif" width="24"> File Structure

```
backend/
├── main.py                # Main Modal app with processing logic
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── .env                  # Your local config (gitignored)
├── README.md             # This file
└── LR-ASD/               # Active speaker detection (git submodule)
    ├── Columbia_test.py  # ASD inference script
    ├── model/            # Model weights
    └── ...
```

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4e6/512.gif" width="24"> Dependencies

### Core Libraries
- `fastapi[standard]` - Web framework
- `modal` - Serverless platform
- `boto3` - AWS S3 client

### AI/ML
- `torch==2.4.1+cu124` - PyTorch with CUDA 12.4
- `torchvision==0.19.1+cu124` - Vision models
- `transformers` - Hugging Face models
- `accelerate` - Model optimization
- `whisperx` - Speech recognition
- `google-generativeai` - Gemini API

### Video Processing
- `ffmpegcv` - GPU-accelerated video I/O
- `opencv-python` - Image processing
- `scenedetect[opencv]` - Scene detection
- `pysubs2` - Subtitle manipulation

### Audio Processing
- `python_speech_features` - Audio features
- `scipy` - Scientific computing

### Utilities
- `pandas` - Data manipulation
- `tqdm` - Progress bars
- `gdown` - Google Drive downloads
- `indic-transliteration` - Hindi transliteration

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2699/512.gif" width="24"> Configuration

### Environment Variables

See [CONFIGURATION.md](../CONFIGURATION.md) for detailed setup.

**Required:**
- `AWS_ACCESS_KEY_ID` - S3 access key
- `AWS_SECRET_ACCESS_KEY` - S3 secret key
- `AWS_REGION` - S3 bucket region
- `S3_BUCKET_NAME` - S3 bucket name
- `GEMINI_API_KEY` - Google Gemini API key
- `AUTH_TOKEN` - API authentication token

### Modal Configuration

Adjust resources in `main.py`:

```python
@app.function(
    gpu="L40S",           # GPU type (L4, A10G, A100, L40S, H100)
    timeout=900,          # Max execution time (seconds)
    memory=16384,         # RAM in MB
    cpu=4,                # CPU cores
)
```

**GPU Options:**
- `L4` - Cheapest, slower ($0.50/hr)
- `A10G` - Good balance ($1.10/hr)
- `L40S` - Fast, good value ($2.50/hr) ← **Current**
- `A100` - Very fast ($4.00/hr)
- `H100` - Fastest ($8.00/hr)

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f41b/512.gif" width="24"> Troubleshooting

### "Model download failed"
**Cause:** First run downloads models (WhisperX, LR-ASD weights)
**Solution:** Wait for download to complete. Models are cached in persistent volume.

### "CUDA out of memory"
**Cause:** GPU memory exhausted
**Solution:**
- Reduce batch size in WhisperX: `batch_size=8`
- Use smaller GPU (A10G) or larger (A100)
- Process shorter videos

### "ffmpeg not found"
**Cause:** ffmpeg not in container image
**Solution:** Ensure `.apt_install(["ffmpeg"])` in image definition

### "S3 upload failed"
**Cause:** Invalid credentials or permissions
**Solution:**
- Verify AWS credentials in Modal secrets
- Check S3 bucket permissions
- Ensure bucket exists in specified region

### "Timeout error"
**Cause:** Processing exceeds 900s limit
**Solution:**
- Increase timeout: `timeout=1800` (30 minutes)
- Process fewer clips per request
- Use faster GPU (L40S → A100)

### "LR-ASD submodule missing"
**Cause:** Submodule not initialized
**Solution:**
```bash
git submodule update --init --recursive
```

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4ca/512.gif" width="24"> Performance

### Processing Times (Approximate)

On L40S GPU for 1-hour podcast:
- **Transcription:** ~3-5 minutes
- **Moment identification:** ~10 seconds
- **Per clip (60s):**
  - ASD: ~2 minutes
  - Vertical video: ~1 minute
  - Subtitles: ~30 seconds
- **Total for 3 clips:** ~12-15 minutes

### Cost Estimation

L40S GPU @ $2.50/hour:
- 15-minute job = $0.625
- 100 jobs/month = $62.50

**Optimization Tips:**
- Use persistent volumes to cache models
- Batch multiple clips in single request
- Consider L4 GPU for non-urgent processing

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f512/512.gif" width="24"> Security

- **API Authentication:** Bearer token required for all requests
- **S3 Access:** Use IAM user with minimal permissions (S3 only)
- **Secrets:** Store in Modal Secrets, never hardcode
- **Temporary Files:** Cleaned up after processing
- **No Data Retention:** Videos not stored on Modal infrastructure

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4dd/512.gif" width="24"> Development

### Local Testing

```bash
# Serve locally
modal serve main.py

# Test with curl
curl -X POST "http://localhost:8000/process_video" \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{"s3_key": "test/video.mp4", "num_clips": 1}'
```

### Debugging

Add print statements (they appear in Modal logs):
```python
print(f"Processing clip {clip_index}...")
```

View logs:
```bash
modal logs clipzz-video-processor
```

### Testing Changes

```bash
# Test locally first
modal serve main.py

# Deploy to production
modal deploy main.py
```

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f680/512.gif" width="24"> Future Improvements

- [ ] Support more video formats (AVI, MOV, MKV)
- [ ] Add configurable subtitle styles
- [ ] Implement progress callbacks (webhooks)
- [ ] Add video quality options (720p, 1080p, 4K)
- [ ] Support multiple languages in subtitles
- [ ] Optimize batch processing (multiple videos at once)
- [ ] Add face recognition for consistent speaker labeling
- [ ] Implement chapter detection for long videos

---

For more information:
- [Architecture Overview](../ARCHITECTURE.md)
- [Configuration Guide](../CONFIGURATION.md)
- [API Documentation](../API.md)
