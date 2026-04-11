# <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f6e0/512.gif" width="32"> Development Guide

Complete guide for developers contributing to Clipzz.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4cb/512.gif" width="24"> Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style & Standards](#code-style--standards)
- [Testing](#testing)
- [Debugging](#debugging)
- [Contributing](#contributing)

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="24"> Prerequisites

### Required Software

- **Node.js** 18+ and npm/pnpm
- **Python** 3.12+
- **Git** for version control
- **Modal CLI** for backend deployment

### Recommended Tools

- **VS Code** with extensions:
  - Prisma
  - ESLint
  - Prettier
  - Python
  - Pylance
- **Postman** or **Thunder Client** for API testing
- **DB Browser for SQLite** for database inspection

### Required Accounts

- [Modal](https://modal.com) - Backend GPU infrastructure
- [Google Cloud Console](https://console.cloud.google.com/) - OAuth & Gemini API
- [AWS](https://aws.amazon.com/) - S3 storage
- [Stripe](https://stripe.com) and/or [Razorpay](https://razorpay.com) - Payments
- [Inngest](https://www.inngest.com/) - Background jobs

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4bb/512.gif" width="24"> Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/clipzz.git
cd clipzz

# Initialize LR-ASD submodule
git submodule update --init --recursive
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Edit .env with your credentials
# See CONFIGURATION.md for detailed setup

# Generate Prisma client
npx prisma generate

# Initialize database
npx prisma db push

# Start development server
npm run dev
```

Frontend runs at [http://localhost:3000](http://localhost:3000)

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token new

# Copy environment file
cp .env.example .env

# Create Modal secrets
modal secret create clipzz-secrets \
  AWS_ACCESS_KEY_ID="your-key" \
  AWS_SECRET_ACCESS_KEY="your-secret" \
  GEMINI_API_KEY="your-gemini-key" \
  AUTH_TOKEN="your-auth-token"

# Serve backend locally
modal serve main.py
```

Backend runs at Modal's local dev URL (shown in terminal)

### 4. Inngest Setup

```bash
cd frontend

# Start Inngest dev server (separate terminal)
npm run inngest-dev
```

Inngest dashboard: [http://localhost:8288](http://localhost:8288)

### 5. Verify Setup

**Frontend checklist:**
- [ ] Can access http://localhost:3000
- [ ] Can sign up with email/password
- [ ] Can sign in with Google OAuth
- [ ] Can access dashboard

**Backend checklist:**
- [ ] Modal serve command runs without errors
- [ ] Can see endpoint URL in terminal
- [ ] Can make test API call with curl

**Inngest checklist:**
- [ ] Dev server running on port 8288
- [ ] Can see events in dashboard

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4c1/512.gif" width="24"> Project Structure

```
clipzz/
├── frontend/                    # Next.js application
│   ├── src/
│   │   ├── app/                # App Router pages
│   │   │   ├── page.tsx        # Landing page
│   │   │   ├── dashboard/      # Main app
│   │   │   │   ├── page.tsx    # Dashboard UI
│   │   │   │   └── billing/    # Payment page
│   │   │   └── api/            # API routes
│   │   │       ├── inngest/    # Inngest endpoint
│   │   │       └── webhooks/   # Payment webhooks
│   │   │
│   │   ├── actions/            # Server actions
│   │   │   ├── auth.ts         # Authentication
│   │   │   ├── s3.ts           # File upload
│   │   │   ├── generate.ts     # Video processing
│   │   │   ├── stripe.ts       # Stripe payments
│   │   │   └── razorpay.ts     # Razorpay payments
│   │   │
│   │   ├── components/         # React components
│   │   │   ├── dashboard-client.tsx
│   │   │   ├── login-form.tsx
│   │   │   ├── signup-form.tsx
│   │   │   └── ui/             # shadcn components
│   │   │
│   │   ├── inngest/            # Background jobs
│   │   │   ├── client.ts       # Inngest client
│   │   │   └── functions.ts    # Job definitions
│   │   │
│   │   ├── lib/                # Utilities
│   │   │   ├── auth.ts         # NextAuth config
│   │   │   └── utils.ts        # Helper functions
│   │   │
│   │   └── server/             # Server-side code
│   │       └── db.ts           # Prisma client
│   │
│   ├── prisma/                 # Database
│   │   └── schema.prisma       # Schema definition
│   │
│   ├── public/                 # Static assets
│   ├── .env.example            # Environment template
│   └── package.json
│
├── backend/                     # Python processing
│   ├── main.py                 # Modal app
│   ├── requirements.txt        # Python deps
│   ├── .env.example            # Environment template
│   └── LR-ASD/                 # Speaker detection (submodule)
│
├── docs/                        # Documentation
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── DATABASE.md
│   ├── CONFIGURATION.md
│   ├── DEVELOPMENT.md (this file)
│   ├── DEPLOYMENT.md
│   ├── SECURITY.md
│   └── USER_GUIDE.md
│
└── .gitignore
```

### Key Files

| File | Purpose |
|------|---------|
| `frontend/src/app/layout.tsx` | Root layout with providers |
| `frontend/src/lib/auth.ts` | NextAuth configuration |
| `frontend/src/inngest/functions.ts` | Video processing workflow |
| `frontend/prisma/schema.prisma` | Database schema |
| `backend/main.py` | Complete video processing pipeline |

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f504/512.gif" width="24"> Development Workflow

### Feature Development

1. **Create feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes:**
   - Write code
   - Add tests (if applicable)
   - Update documentation

3. **Test locally:**
```bash
# Frontend
npm run check        # Type check
npm run lint         # Lint
npm run build        # Test build

# Backend
modal serve main.py  # Test locally
```

4. **Commit changes:**
```bash
git add .
git commit -m "feat: add feature description"
```

5. **Push and create PR:**
```bash
git push origin feature/your-feature-name
# Create Pull Request on GitHub
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code restructure
- `test:` Tests
- `chore:` Maintenance

**Examples:**
```bash
feat(dashboard): add clip filtering
fix(auth): resolve Google OAuth redirect issue
docs(api): update endpoint documentation
refactor(video): improve subtitle generation
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2728/512.gif" width="24"> Code Style & Standards

### TypeScript/JavaScript

**Formatter:** Prettier
**Linter:** ESLint

```bash
# Run formatter
npm run format

# Run linter
npm run lint

# Fix auto-fixable issues
npm run lint --fix
```

**Naming conventions:**
- `camelCase` for variables and functions
- `PascalCase` for components and types
- `UPPER_SNAKE_CASE` for constants
- Descriptive names (no single letters except loop counters)

**Example:**
```typescript
// Good
const userCredits = await getUserCredits(userId);
const MAX_CLIPS_PER_VIDEO = 10;

function processVideo(uploadId: string) { ... }

interface VideoProcessRequest {
  s3Key: string;
  numClips: number;
}

// Bad
const x = await getC(u);
const max = 10;

function pv(id: string) { ... }

interface vpr {
  s: string;
  n: number;
}
```

### Python

**Formatter:** Black (recommended)
**Linter:** Ruff (recommended)

```bash
# Format code
black backend/

# Lint code
ruff check backend/
```

**Naming conventions:**
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Docstrings for all public functions

**Example:**
```python
# Good
def process_video_clip(video_path: str, start_time: float) -> str:
    """
    Process a video clip.

    Args:
        video_path: Path to video file
        start_time: Start timestamp in seconds

    Returns:
        Path to processed clip
    """
    MAX_DURATION = 60
    clip_duration = calculate_duration(start_time)
    return create_clip(video_path, clip_duration)

# Bad
def pvc(vp: str, st: float) -> str:
    md = 60
    cd = calc_dur(st)
    return cc(vp, cd)
```

### React Components

**Prefer:**
- Server Components by default
- Client Components only when needed
- Composition over prop drilling
- Custom hooks for reusable logic

**Example:**
```tsx
// Good: Server Component
export default async function DashboardPage() {
  const uploads = await getUploads();
  return <DashboardClient uploads={uploads} />;
}

// Good: Client Component when needed
"use client";
export function DashboardClient({ uploads }) {
  const [filter, setFilter] = useState("");
  // ... interactive logic
}

// Bad: Unnecessary client component
"use client";
export default function DashboardPage() {
  const [uploads, setUploads] = useState([]);
  useEffect(() => { fetchUploads(); }, []);
  // Should be server component!
}
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f9ea/512.gif" width="24"> Testing

### Current Status

<img src="https://fonts.gstatic.com/s/e/notoemoji/latest/26a0/512.gif" width="16"> **No test suite currently implemented.**

### Recommended Testing Strategy

#### Frontend Testing

**Tools:** Vitest + React Testing Library

```bash
# Install testing dependencies
npm install -D vitest @testing-library/react @testing-library/jest-dom
```

**Test structure:**
```
src/
├── actions/
│   ├── auth.ts
│   └── auth.test.ts
├── components/
│   ├── dashboard-client.tsx
│   └── dashboard-client.test.tsx
└── lib/
    ├── utils.ts
    └── utils.test.ts
```

**Example test:**
```typescript
// src/actions/auth.test.ts
import { describe, it, expect } from 'vitest';
import { hashPassword, verifyPassword } from './auth';

describe('Password hashing', () => {
  it('should hash password', async () => {
    const password = 'test123';
    const hashed = await hashPassword(password);
    expect(hashed).not.toBe(password);
  });

  it('should verify correct password', async () => {
    const password = 'test123';
    const hashed = await hashPassword(password);
    const isValid = await verifyPassword(password, hashed);
    expect(isValid).toBe(true);
  });
});
```

#### Backend Testing

**Tools:** pytest

```bash
# Install testing dependencies
pip install pytest pytest-asyncio
```

**Test structure:**
```
backend/
├── main.py
├── test_main.py
└── tests/
    ├── test_transcription.py
    ├── test_clip_processing.py
    └── test_subtitles.py
```

**Example test:**
```python
# backend/test_main.py
import pytest
from main import is_hindi, transliterate_hindi_to_english

def test_is_hindi():
    assert is_hindi("नमस्ते") == True
    assert is_hindi("Hello") == False

def test_transliterate():
    result = transliterate_hindi_to_english("नमस्ते")
    assert "namaste" in result.lower()
```

#### Integration Testing

**Webhook testing:**
```bash
# Stripe webhook
stripe listen --forward-to localhost:3000/api/webhooks/stripe

# Test payment
stripe trigger checkout.session.completed
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f41b/512.gif" width="24"> Debugging

### Frontend Debugging

**VS Code launch.json:**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Next.js: debug server-side",
      "type": "node-terminal",
      "request": "launch",
      "command": "npm run dev"
    },
    {
      "name": "Next.js: debug client-side",
      "type": "chrome",
      "request": "launch",
      "url": "http://localhost:3000"
    }
  ]
}
```

**Common issues:**

| Issue | Solution |
|-------|----------|
| Prisma client out of sync | Run `npx prisma generate` |
| NextAuth session undefined | Check `AUTH_SECRET` in .env |
| S3 upload fails | Verify CORS configuration |
| Can't connect to Inngest | Ensure `npm run inngest-dev` is running |

### Backend Debugging

**Modal logs:**
```bash
# View live logs
modal logs clipzz-video-processor

# View logs for specific run
modal logs clipzz-video-processor --run-id abc123
```

**Add debug prints:**
```python
# Logs appear in Modal dashboard
print(f"Processing clip {clip_index}...")
print(f"Transcript segments: {len(segments)}")
```

**Common issues:**

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size or use larger GPU |
| Model download timeout | Wait longer (first run downloads models) |
| S3 access denied | Check AWS credentials in Modal secrets |
| ffmpeg not found | Verify container image includes ffmpeg |

### Database Debugging

**Prisma Studio:**
```bash
cd frontend
npx prisma studio
```

Opens GUI at [http://localhost:5555](http://localhost:5555)

**Direct SQL queries:**
```bash
# SQLite
sqlite3 frontend/prisma/dev.db

# View tables
.tables

# Query users
SELECT * FROM User;
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f91d/512.gif" width="24"> Contributing

### Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Submit a pull request

### Pull Request Guidelines

**Before submitting:**
- [ ] Code follows style guidelines
- [ ] All tests pass (when implemented)
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts

**PR description should include:**
- Summary of changes
- Motivation/context
- Screenshots (for UI changes)
- Testing instructions
- Breaking changes (if any)

**Example PR description:**
```markdown
## Summary
Add clip filtering by status in dashboard

## Motivation
Users requested ability to filter clips by processing status

## Changes
- Add status filter dropdown to dashboard
- Implement server-side filtering
- Add URL search params for persistence

## Testing
1. Upload and process video
2. Use filter dropdown
3. Verify URL updates
4. Refresh page - filter persists

## Screenshots
![Filter dropdown](link-to-image)
```

### Code Review Process

1. Automated checks run (lint, type check, build)
2. Maintainer reviews code
3. Feedback addressed
4. Approved and merged

**Review criteria:**
- Code quality and readability
- Performance implications
- Security considerations
- Documentation completeness

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4da/512.gif" width="24"> Useful Commands

### Frontend

```bash
# Development
npm run dev              # Start dev server
npm run build            # Production build
npm run start            # Start production server
npm run check            # Type check
npm run lint             # Lint code
npm run format           # Format with Prettier

# Database
npx prisma studio        # Database GUI
npx prisma generate      # Generate client
npx prisma db push       # Sync schema (dev)
npx prisma migrate dev   # Create migration
npx prisma migrate deploy # Apply migrations (prod)

# Inngest
npm run inngest-dev      # Start Inngest dev server
```

### Backend

```bash
# Development
modal serve main.py      # Run locally
modal deploy main.py     # Deploy to production
modal logs clipzz        # View logs
modal secret list        # List secrets

# Python
pip install -r requirements.txt  # Install deps
pip freeze > requirements.txt    # Update deps
black .                          # Format code
ruff check .                     # Lint code
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4ac/512.gif" width="24"> Getting Help

- **Documentation:** Check [README.md](README.md) and other docs
- **Issues:** Search [GitHub Issues](https://github.com/yourusername/clipzz/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/clipzz/discussions)
- **Discord:** (if available)

---

<div align="center">
  <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2764/512.gif" width="20"> Happy coding!
</div>
