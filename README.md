
# <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f3a5/512.gif" width="32"> Clipzz - AI-Powered Podcast Clip Generator

> Transform your long-form podcasts into viral-ready vertical clips with AI

[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org/)
[![Modal](https://img.shields.io/badge/Modal-Serverless-purple)](https://modal.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Clipzz automatically converts your long-form podcast videos into engaging short-form vertical clips optimized for TikTok, Instagram Reels, and YouTube Shorts. Using cutting-edge AI for speaker detection, transcription, and moment identification, Clipzz makes content creation effortless.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2728/512.gif" width="24"> Features

### <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f916/512.gif" width="20"> AI-Powered Processing

- **Automatic Transcription** - WhisperX with multi-language support (English, Hindi)
- **Smart Moment Detection** - Google Gemini 2.5 Pro identifies engaging Q&A segments
- **Active Speaker Detection** - LR-ASD model tracks and focuses on speaking persons
- **Intelligent Cropping** - Dynamic camera framing that follows the active speaker

### <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f3ac/512.gif" width="20"> Video Processing

- **Vertical Format** - Converts horizontal videos to 1080x1920 (perfect for social media)
- **Animated Subtitles** - Karaoke-style word-by-word highlighting
- **Smart Background** - Blurred letterboxing when no speaker detected
- **High Quality** - GPU-accelerated processing with FFmpeg

### <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4b3/512.gif" width="20"> Business Features

- **Credit-Based System** - Pay only for what you process
- **Multiple Payment Options** - Stripe (global) and Razorpay (India)
- **User Dashboard** - Track processing status and manage clips
- **Secure Storage** - AWS S3 with presigned URLs

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4bb/512.gif" width="24"> Tech Stack

### Frontend
- **Framework:** Next.js 15 (App Router) + React 19 + TypeScript
- **Styling:** Tailwind CSS 4 + Shadcn UI + Radix UI
- **Auth:** NextAuth.js v5 (Google OAuth + Credentials)
- **Database:** Prisma ORM with SQLite (dev) / PostgreSQL (prod)
- **Background Jobs:** Inngest
- **Storage:** AWS S3
- **Payments:** Stripe + Razorpay

### Backend
- **Framework:** FastAPI
- **Infrastructure:** Modal (Serverless GPU - L40S)
- **AI Models:**
  - WhisperX (large-v2) - Speech-to-text
  - Google Gemini 2.5 Pro - Content analysis
  - LR-ASD - Active speaker detection
- **Video Processing:** PyTorch, OpenCV, FFmpeg
- **Languages:** Python 3.12

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f680/512.gif" width="24"> Quick Start

### Prerequisites

- Node.js 18+ and npm/pnpm
- Python 3.12+
- AWS account with S3 bucket
- Modal account
- Google Gemini API key
- Stripe or Razorpay account

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clipzz.git
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
# Frontend
cp frontend/.env.example frontend/.env

# Backend
cp backend/.env.example backend/.env
```

2. Configure environment variables (see [CONFIGURATION.md](CONFIGURATION.md) for details)

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

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4da/512.gif" width="24"> Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - System design and data flow
- **[API Documentation](docs/API.md)** - API endpoints and server actions
- **[Backend README](backend/README.md)** - Backend processing pipeline

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4a1/512.gif" width="24"> How It Works

1. **Upload** - User uploads podcast video through dashboard
2. **Transcribe** - WhisperX transcribes audio with word-level timestamps
3. **Analyze** - Gemini AI identifies the most engaging 30-60 second moments
4. **Detect** - LR-ASD tracks faces and determines active speaker per frame
5. **Process** - For each clip:
   - Extract video segment
   - Create vertical format with smart cropping
   - Generate karaoke-style subtitles
   - Burn subtitles into video
6. **Store** - Upload clips to S3
7. **Deliver** - User downloads clips from dashboard

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4b0/512.gif" width="24"> Pricing

| Pack | Credits | Price (USD) | Price (INR) |
|------|---------|-------------|-------------|
| Small | 50 | $10 | ₹830 |
| Medium | 150 | $25 | ₹2,075 |
| Large | 500 | $70 | ₹5,810 |

- 1 Credit = 1 Generated Clip
- New users get 10 free credits
- Credits never expire

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f6e0/512.gif" width="24"> Project Structure

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
├── ARCHITECTURE.md
├── CONFIGURATION.md
└── README.md
```

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f91d/512.gif" width="24"> Contributing

We welcome contributions! Please see [DEVELOPMENT.md](DEVELOPMENT.md) for:

- Setting up your development environment
- Code style guidelines
- Testing requirements
- Pull request process

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f512/512.gif" width="24"> Security

For security concerns, please review [SECURITY.md](SECURITY.md) and report vulnerabilities privately.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4dc/512.gif" width="24"> License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4ac/512.gif" width="24"> Support

- **Documentation:** Check the docs folder
- **Issues:** [GitHub Issues](https://github.com/yourusername/clipzz/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/clipzz/discussions)

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f64f/512.gif" width="24"> Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) - Speech recognition
- [LR-ASD](https://github.com/Junhua-Liao/Light-ASD) - Active speaker detection
- [Google Gemini](https://ai.google.dev/) - AI content analysis
- [Modal](https://modal.com/) - Serverless GPU infrastructure


