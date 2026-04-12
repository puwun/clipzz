<div align="center">
  <img src="frontend/public/android-chrome-512x512.png" alt="Clipzz Logo" width="120" />
  <h1>Clipzz</h1>
  <p><strong>AI-Powered Podcast Clip Generator</strong></p>
  <p>Transform your long-form podcasts into viral-ready vertical clips with AI</p>

  [![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org/)
  [![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org/)
  [![Modal](https://img.shields.io/badge/Modal-Serverless_GPU-purple)](https://modal.com/)
  [![Live](https://img.shields.io/badge/Live-clipzzz.vercel.app-success)](https://clipzzz.vercel.app)
</div>

---

Clipzz automatically converts your long-form podcast videos into engaging short-form vertical clips optimized for TikTok, Instagram Reels, and YouTube Shorts. Using cutting-edge AI for speaker detection, transcription, and moment identification, Clipzz makes content creation effortless.

## ✨ Features

### 🤖 AI-Powered Processing

- **Automatic Transcription** — WhisperX with multi-language support (English, Hindi)
- **Smart Moment Detection** — Google Gemini 2.5 Pro identifies engaging Q&A segments
- **Active Speaker Detection** — LR-ASD model tracks and focuses on speaking persons
- **Intelligent Cropping** — Dynamic camera framing that follows the active speaker

### 🎬 Video Processing

- **Vertical Format** — Converts horizontal videos to 1080×1920 (perfect for social media)
- **Animated Subtitles** — Karaoke-style word-by-word highlighting
- **Smart Background** — Blurred letterboxing when no speaker detected
- **High Quality** — GPU-accelerated processing with FFmpeg

### 💳 Business Features

- **Credit-Based System** — Pay only for what you process
- **Multiple Payment Options** — Stripe (global) and Razorpay (India)
- **User Dashboard** — Track processing status and manage clips
- **Secure Storage** — AWS S3 with presigned URLs

## 💻 Tech Stack

### Frontend
- **Framework:** Next.js 16 (App Router) + React 19 + TypeScript
- **Styling:** Tailwind CSS 4 + Shadcn UI + Radix UI
- **Auth:** NextAuth.js v5 (Google OAuth + Credentials)
- **Database:** Prisma ORM with PostgreSQL (Neon)
- **Background Jobs:** Inngest
- **Storage:** AWS S3
- **Payments:** Stripe + Razorpay

### Backend
- **Framework:** FastAPI
- **Infrastructure:** Modal (Serverless GPU — L40S)
- **AI Models:**
  - WhisperX (large-v2) — Speech-to-text
  - Google Gemini 2.5 Pro — Content analysis
  - LR-ASD — Active speaker detection
- **Video Processing:** PyTorch, OpenCV, FFmpeg
- **Languages:** Python 3.12

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.12+
- AWS account with S3 bucket
- Modal account
- Google Gemini API key
- Stripe or Razorpay account

### Installation

```bash
# Clone the repository
git clone https://github.com/puwun/clipzz.git
cd clipzz

# Install frontend dependencies
cd frontend
npm install

# Install backend dependencies
cd ../backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

1. Copy environment files:
```bash
cp frontend/.env.example frontend/.env
cp backend/.env.example backend/.env
```

2. Fill in environment variables (see `.env.example` for details)

3. Initialize database:
```bash
cd frontend
npx prisma db push
```

### Running Locally

```bash
# Terminal 1: Frontend
cd frontend
npm run dev

# Terminal 2: Backend (Modal local)
cd backend
modal serve main.py

# Terminal 3: Inngest Dev Server
cd frontend
npm run inngest-dev
```

Visit [http://localhost:3000](http://localhost:3000) to see the app.

## 📖 Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — System design and data flow
- **[API Documentation](docs/API.md)** — API endpoints and server actions
- **[Backend README](backend/README.md)** — Backend processing pipeline

## 💡 How It Works

1. **Upload** — User uploads podcast video through dashboard
2. **Transcribe** — WhisperX transcribes audio with word-level timestamps
3. **Analyze** — Gemini AI identifies the most engaging 30–60 second moments
4. **Detect** — LR-ASD tracks faces and determines active speaker per frame
5. **Process** — For each clip:
   - Extract video segment
   - Create vertical format with smart cropping
   - Generate karaoke-style subtitles
   - Burn subtitles into video
6. **Store** — Upload clips to S3
7. **Deliver** — User downloads clips from dashboard

## 💰 Pricing

| Pack | Credits | Price (USD) | Price (INR) |
|------|---------|-------------|-------------|
| Small | 50 | $10 | ₹830 |
| Medium | 150 | $25 | ₹2,075 |
| Large | 500 | $70 | ₹5,810 |

- 1 Credit = 1 Generated Clip
- New users get **10 free credits**
- Credits never expire

## 🗂️ Project Structure

```
clipzz/
├── frontend/              # Next.js application
│   ├── src/
│   │   ├── actions/       # Server actions
│   │   ├── app/           # App router pages
│   │   ├── components/    # React components
│   │   ├── inngest/       # Background jobs
│   │   ├── lib/           # Utilities
│   │   └── server/        # Server-side code
│   ├── prisma/            # Database schema
│   └── package.json
├── backend/               # Python processing pipeline
│   ├── LR-ASD/           # Active speaker detection
│   ├── main.py           # Main Modal app
│   └── requirements.txt
├── docs/                  # Documentation
└── README.md
```

## 🙏 Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) — Speech recognition
- [LR-ASD](https://github.com/Junhua-Liao/Light-ASD) — Active speaker detection
- [Google Gemini](https://ai.google.dev/) — AI content analysis
- [Modal](https://modal.com/) — Serverless GPU infrastructure
