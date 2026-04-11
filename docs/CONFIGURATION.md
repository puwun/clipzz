# <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2699/512.gif" width="32"> Configuration Guide

Complete guide to configuring Clipzz for development and production environments.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4cb/512.gif" width="24"> Table of Contents

- [Environment Files](#environment-files)
- [Frontend Configuration](#frontend-configuration)
- [Backend Configuration](#backend-configuration)
- [External Services Setup](#external-services-setup)
- [Development vs Production](#development-vs-production)

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4c1/512.gif" width="24"> Environment Files

### Frontend (.env)
Location: `frontend/.env`

```bash
# ============================================
# AUTHENTICATION
# ============================================

# NextAuth.js Secret (REQUIRED)
# Generate with: openssl rand -base64 32
AUTH_SECRET="your-secret-key-here"

# Google OAuth (REQUIRED for Google Sign-In)
GOOGLE_CLIENT_ID="your-google-client-id"
GOOGLE_CLIENT_SECRET="your-google-client-secret"

# ============================================
# DATABASE
# ============================================

# SQLite (Development)
DATABASE_URL="file:./dev.db"

# PostgreSQL (Production - Recommended)
# DATABASE_URL="postgresql://user:password@host:5432/clipzz?schema=public"

# ============================================
# AWS S3 STORAGE
# ============================================

# AWS Credentials (REQUIRED)
AWS_ACCESS_KEY_ID="your-aws-access-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
AWS_REGION="us-east-1"
S3_BUCKET_NAME="clipzz-videos"

# ============================================
# MODAL BACKEND
# ============================================

# Modal API Endpoint (REQUIRED)
PROCESS_VIDEO_ENDPOINT="https://your-app--clipzz-video-processor-process-video.modal.run"
PROCESS_VIDEO_ENDPOINT_AUTH="your-auth-token"

# ============================================
# STRIPE PAYMENT (Global)
# ============================================

# Stripe Keys
STRIPE_SECRET_KEY="sk_test_..."
NEXT_PUBLIC_STRIPE_PUB_KEY="pk_test_..."

# Stripe Product Price IDs
STRIPE_SMALL_CREDIT_PACK="price_xxxxxxxxxxxxx"
STRIPE_MEDIUM_CREDIT_PACK="price_xxxxxxxxxxxxx"
STRIPE_LARGE_CREDIT_PACK="price_xxxxxxxxxxxxx"

# Stripe Webhook Secret
STRIPE_WEBHOOK_SECRET="whsec_..."

# ============================================
# RAZORPAY PAYMENT (India)
# ============================================

# Razorpay Keys
RAZORPAY_KEY_ID="rzp_test_..."
RAZORPAY_KEY_SECRET="your-razorpay-secret"

# Razorpay Product IDs (Optional - used for validation)
RAZORPAY_SMALL_CREDIT_PACK="small"
RAZORPAY_MEDIUM_CREDIT_PACK="medium"
RAZORPAY_LARGE_CREDIT_PACK="large"

# Razorpay Webhook Secret
RAZORPAY_WEBHOOK_SECRET="your-webhook-secret"

# ============================================
# APPLICATION
# ============================================

# Base URL (for webhooks and redirects)
BASE_URL="http://localhost:3000"  # Development
# BASE_URL="https://clipzz.com"   # Production
```

### Backend (.env)
Location: `backend/.env`

```bash
# ============================================
# AWS S3 STORAGE
# ============================================

AWS_ACCESS_KEY_ID="your-aws-access-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
AWS_REGION="us-east-1"
S3_BUCKET_NAME="clipzz-videos"

# ============================================
# AI MODELS
# ============================================

# Google Gemini API Key (REQUIRED)
GEMINI_API_KEY="your-gemini-api-key"

# ============================================
# MODAL CONFIGURATION
# ============================================

# Authentication Token for API Endpoint (REQUIRED)
# Generate a strong random token
AUTH_TOKEN="your-secure-auth-token"

# Note: Modal secrets should be set via Modal CLI:
# modal secret create clipzz-secrets \
#   AWS_ACCESS_KEY_ID=xxx \
#   AWS_SECRET_ACCESS_KEY=xxx \
#   GEMINI_API_KEY=xxx \
#   AUTH_TOKEN=xxx
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4e6/512.gif" width="24"> Frontend Configuration

### 1. Authentication (NextAuth.js)

#### Generate AUTH_SECRET
```bash
openssl rand -base64 32
```
Add to `frontend/.env`:
```bash
AUTH_SECRET="generated-secret-here"
```

#### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
5. Application type: **Web application**
6. Authorized redirect URIs:
   - Development: `http://localhost:3000/api/auth/callback/google`
   - Production: `https://yourdomain.com/api/auth/callback/google`
7. Copy **Client ID** and **Client Secret**

```bash
GOOGLE_CLIENT_ID="123456789-abcdefg.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET="GOCSPX-xxxxxxxxxxxxx"
```

### 2. Database Setup

#### SQLite (Development)
Already configured by default:
```bash
DATABASE_URL="file:./dev.db"
```

Initialize database:
```bash
cd frontend
npx prisma db push
npx prisma generate
```

#### PostgreSQL (Production)

1. Create PostgreSQL database
2. Update `DATABASE_URL`:
```bash
DATABASE_URL="postgresql://username:password@host:5432/database?schema=public"
```

3. Run migrations:
```bash
npx prisma migrate deploy
npx prisma generate
```

### 3. AWS S3 Storage

#### Create S3 Bucket

1. Log in to [AWS Console](https://console.aws.amazon.com/)
2. Go to **S3** → **Create bucket**
3. Bucket name: `clipzz-videos` (or your choice)
4. Region: `us-east-1` (or preferred)
5. **Block all public access:** OFF (we use presigned URLs)
6. Create bucket

#### Configure CORS

Add CORS policy to your bucket:
```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST"],
    "AllowedOrigins": ["http://localhost:3000", "https://yourdomain.com"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```

#### Create IAM User

1. Go to **IAM** → **Users** → **Add user**
2. User name: `clipzz-s3-user`
3. Attach policy: `AmazonS3FullAccess` (or create custom policy)
4. Create access key → **Application running outside AWS**
5. Copy **Access Key ID** and **Secret Access Key**

```bash
AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
AWS_REGION="us-east-1"
S3_BUCKET_NAME="clipzz-videos"
```

### 4. Payment Gateways

#### Stripe (Global)

1. Sign up at [stripe.com](https://stripe.com)
2. Get API keys from [Dashboard → Developers → API keys](https://dashboard.stripe.com/apikeys)

```bash
STRIPE_SECRET_KEY="sk_test_51xxxxxxxxxxxxx"
NEXT_PUBLIC_STRIPE_PUB_KEY="pk_test_51xxxxxxxxxxxxx"
```

3. Create products and prices:
   - Go to **Products** → **Add product**
   - Create three products:
     - Small Pack: $10 (50 credits)
     - Medium Pack: $25 (150 credits)
     - Large Pack: $70 (500 credits)
   - Copy each price ID (starts with `price_`)

```bash
STRIPE_SMALL_CREDIT_PACK="price_1xxxxxxxxxxxxx"
STRIPE_MEDIUM_CREDIT_PACK="price_1xxxxxxxxxxxxx"
STRIPE_LARGE_CREDIT_PACK="price_1xxxxxxxxxxxxx"
```

4. Configure webhook:
   - Go to **Developers** → **Webhooks** → **Add endpoint**
   - Endpoint URL: `https://yourdomain.com/api/webhooks/stripe`
   - Events to listen: `checkout.session.completed`
   - Copy signing secret

```bash
STRIPE_WEBHOOK_SECRET="whsec_xxxxxxxxxxxxx"
```

5. Test webhook locally:
```bash
# Install Stripe CLI
stripe listen --forward-to localhost:3000/api/webhooks/stripe
```

#### Razorpay (India)

1. Sign up at [razorpay.com](https://razorpay.com)
2. Get API keys from [Dashboard → Settings → API Keys](https://dashboard.razorpay.com/app/keys)

```bash
RAZORPAY_KEY_ID="rzp_test_xxxxxxxxxxxxx"
RAZORPAY_KEY_SECRET="xxxxxxxxxxxxx"
```

3. Configure webhook:
   - Go to **Settings** → **Webhooks** → **Add webhook**
   - Webhook URL: `https://yourdomain.com/api/webhooks/razorpay`
   - Active Events: `order.paid`
   - Copy webhook secret

```bash
RAZORPAY_WEBHOOK_SECRET="xxxxxxxxxxxxx"
```

4. Credit pack IDs (optional, used in code):
```bash
RAZORPAY_SMALL_CREDIT_PACK="small"
RAZORPAY_MEDIUM_CREDIT_PACK="medium"
RAZORPAY_LARGE_CREDIT_PACK="large"
```

Prices are hardcoded in `frontend/src/actions/razorpay.ts`:
- Small: ₹830 (83000 paise)
- Medium: ₹2075 (207500 paise)
- Large: ₹5810 (581000 paise)

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f9ea/512.gif" width="24"> Backend Configuration

### 1. Modal Setup

#### Install Modal CLI
```bash
pip install modal
```

#### Authenticate
```bash
modal token new
```
This opens browser for authentication and stores token locally.

#### Create Modal Secrets

Modal uses secrets for environment variables. Create a secret named `clipzz-secrets`:

```bash
modal secret create clipzz-secrets \
  AWS_ACCESS_KEY_ID="your-aws-key" \
  AWS_SECRET_ACCESS_KEY="your-aws-secret" \
  GEMINI_API_KEY="your-gemini-key" \
  AUTH_TOKEN="your-auth-token"
```

Update `backend/main.py` to reference this secret:
```python
@app.function(
    secrets=[modal.Secret.from_name("clipzz-secrets")],
    # ... other config
)
```

#### Generate AUTH_TOKEN
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Use this token in both:
- Modal secret: `AUTH_TOKEN`
- Frontend env: `PROCESS_VIDEO_ENDPOINT_AUTH`

### 2. Google Gemini API

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **Create API Key**
3. Copy the key

```bash
GEMINI_API_KEY="AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

### 3. Deploy Backend to Modal

```bash
cd backend
modal deploy main.py
```

This outputs the endpoint URL:
```
✓ Created app clipzz-video-processor
Function URL: https://your-app--clipzz-video-processor-process-video.modal.run
```

Copy this URL to frontend `.env`:
```bash
PROCESS_VIDEO_ENDPOINT="https://your-app--clipzz-video-processor-process-video.modal.run"
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f310/512.gif" width="24"> External Services Setup

### Inngest

1. Sign up at [inngest.com](https://www.inngest.com/)
2. Create a new app
3. No additional configuration needed (auto-discovered via `/api/inngest` route)

For local development:
```bash
cd frontend
npm run inngest-dev
```
This starts Inngest Dev Server at http://localhost:8288

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2696/512.gif" width="24"> Development vs Production

### Development Configuration

```bash
# Frontend
BASE_URL="http://localhost:3000"
DATABASE_URL="file:./dev.db"
STRIPE_SECRET_KEY="sk_test_..."
RAZORPAY_KEY_ID="rzp_test_..."

# Backend (Modal)
modal serve main.py  # Local development mode
```

### Production Configuration

```bash
# Frontend
BASE_URL="https://yourdomain.com"
DATABASE_URL="postgresql://..."
STRIPE_SECRET_KEY="sk_live_..."
RAZORPAY_KEY_ID="rzp_live_..."

# Backend (Modal)
modal deploy main.py  # Production deployment
```

### Production Checklist

- [ ] Switch to PostgreSQL database
- [ ] Use production Stripe/Razorpay keys
- [ ] Update `BASE_URL` to production domain
- [ ] Configure production webhooks
- [ ] Enable HTTPS
- [ ] Set up CDN for S3 (CloudFront)
- [ ] Configure monitoring (Sentry, LogDNA)
- [ ] Set up backup strategy for database
- [ ] Review S3 bucket permissions
- [ ] Rotate `AUTH_SECRET` and `AUTH_TOKEN`

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f6e1/512.gif" width="24"> Security Best Practices

### Environment Variables

1. **Never commit `.env` files** to git (already in `.gitignore`)
2. **Use strong secrets:**
   ```bash
   # Good (32+ characters, random)
   AUTH_SECRET="aB3$xY9#mK2@pL7&nQ4%rT8!vW6^sZ1*"

   # Bad (weak, predictable)
   AUTH_SECRET="mysecret123"
   ```
3. **Rotate secrets regularly** (every 90 days for production)
4. **Use different secrets** for dev/staging/production
5. **Limit IAM permissions** (principle of least privilege)

### API Keys

- Store in environment variables, never hardcode
- Use separate keys for different environments
- Monitor API usage for anomalies
- Set up billing alerts

### Webhooks

- Always verify webhook signatures
- Use HTTPS endpoints only
- Implement replay attack protection
- Log all webhook events

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f41b/512.gif" width="24"> Troubleshooting

### "AUTH_SECRET not set"
**Solution:** Generate and add to `.env`:
```bash
openssl rand -base64 32
```

### "Cannot connect to database"
**Solution:** Check `DATABASE_URL` format and database is running:
```bash
# SQLite
ls frontend/prisma/dev.db

# PostgreSQL
psql $DATABASE_URL -c "SELECT 1;"
```

### "S3 Access Denied"
**Solution:**
1. Check IAM user has S3 permissions
2. Verify bucket name and region match
3. Check CORS configuration

### "Modal function timeout"
**Solution:** Increase timeout in `main.py`:
```python
@app.function(timeout=1200)  # 20 minutes
```

### "Webhook signature verification failed"
**Solution:**
1. Verify webhook secret matches
2. Check webhook endpoint URL is correct
3. Ensure raw body is used for verification (no parsing)

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4dd/512.gif" width="24"> Configuration Templates

### Complete .env Template (Frontend)
Download: [`frontend/.env.example`](frontend/.env.example)

### Complete .env Template (Backend)
Download: [`backend/.env.example`](backend/.env.example)

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4de/512.gif" width="24"> Need Help?

- Check [DEVELOPMENT.md](DEVELOPMENT.md) for setup instructions
- Review [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- See [SECURITY.md](SECURITY.md) for security guidelines
- Open an issue on GitHub

---

<div align="center">
  <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="20"> Configuration complete! Ready to run Clipzz.
</div>
