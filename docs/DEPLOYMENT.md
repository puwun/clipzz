# <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f680/512.gif" width="32"> Deployment Guide

Complete guide for deploying Clipzz to production.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4cb/512.gif" width="24"> Pre-Deployment Checklist

Before deploying to production, ensure:

- [ ] All tests pass (when implemented)
- [ ] Environment variables configured
- [ ] Database migrations ready
- [ ] S3 bucket created and configured
- [ ] Payment gateways configured
- [ ] Domain name purchased
- [ ] SSL certificates ready
- [ ] Monitoring setup
- [ ] Backup strategy defined

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4e6/512.gif" width="24"> Infrastructure Overview

```
┌─────────────────────────────────────────┐
│           CloudFlare CDN                │
│         (Optional - for assets)         │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           Vercel Edge Network           │
│     (Frontend - Next.js deployment)     │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────┐         ┌─────▼──────┐
│ PostgreSQL │         │ Inngest    │
│ (Database) │         │ (Jobs)     │
└────────────┘         └─────┬──────┘
                             │
                       ┌─────▼──────┐
                       │   Modal    │
                       │ (Backend)  │
                       └─────┬──────┘
                             │
                       ┌─────▼──────┐
                       │   AWS S3   │
                       │ (Storage)  │
                       └────────────┘
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f5c4/512.gif" width="24"> Database Deployment

### Migrate to PostgreSQL

**Why?**
- SQLite not suitable for production
- Better concurrency handling
- Connection pooling
- Scalability

### Recommended Providers

#### Option 1: Vercel Postgres (Easiest)
```bash
# Install Vercel CLI
npm i -g vercel

# Create database
vercel postgres create clipzz-db

# Get connection string
vercel postgres show clipzz-db

# Update .env
DATABASE_URL="postgres://..."
```

#### Option 2: Supabase (Generous free tier)
1. Go to [supabase.com](https://supabase.com)
2. Create project
3. Get connection string from Settings → Database
4. Update `DATABASE_URL`

#### Option 3: Railway (Simple)
1. Go to [railway.app](https://railway.app)
2. New Project → PostgreSQL
3. Copy connection string
4. Update `DATABASE_URL`

### Apply Migrations

```bash
cd frontend

# Update schema.prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

# Uncomment @db.Text annotations in Account model
model Account {
  refresh_token String? @db.Text
  access_token  String? @db.Text
  id_token      String? @db.Text
}

# Generate migration
npx prisma migrate deploy

# Verify
npx prisma studio
```

### Connection Pooling

```bash
# Use connection pooler URL (recommended for serverless)
DATABASE_URL="postgres://user:pass@host:5432/db?pgbouncer=true&connection_limit=1"
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2699/512.gif" width="24"> Backend Deployment (Modal)

### 1. Deploy to Modal

```bash
cd backend

# Ensure Modal CLI installed
pip install modal

# Authenticate (if not done)
modal token new

# Deploy to production
modal deploy main.py
```

**Output:**
```
✓ Created app clipzz-video-processor
Function URL: https://your-app--clipzz-video-processor-process-video.modal.run
```

### 2. Update Frontend Environment

Copy the production URL to frontend `.env`:

```bash
# frontend/.env (production)
PROCESS_VIDEO_ENDPOINT="https://your-app--clipzz-video-processor-process-video.modal.run"
PROCESS_VIDEO_ENDPOINT_AUTH="your-production-auth-token"
```

### 3. Configure Modal Secrets

```bash
# Production secrets
modal secret create clipzz-secrets-prod \
  AWS_ACCESS_KEY_ID="prod-key" \
  AWS_SECRET_ACCESS_KEY="prod-secret" \
  GEMINI_API_KEY="prod-gemini-key" \
  AUTH_TOKEN="prod-auth-token"
```

**Update main.py:**
```python
@app.function(
    secrets=[modal.Secret.from_name("clipzz-secrets-prod")],
    # ... rest of config
)
```

### 4. Monitor Modal

```bash
# View logs
modal logs clipzz-video-processor

# View app status
modal app list
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f310/512.gif" width="24"> Frontend Deployment (Vercel)

### 1. Connect Repository

1. Go to [vercel.com](https://vercel.com)
2. Click **"Add New Project"**
3. Import Git Repository
4. Select `clipzz` repository

### 2. Configure Build Settings

**Framework Preset:** Next.js
**Root Directory:** `frontend`
**Build Command:** `npm run build`
**Output Directory:** `.next`
**Install Command:** `npm install`

### 3. Environment Variables

Add all variables from `.env` to Vercel:

```
AUTH_SECRET=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
DATABASE_URL=postgresql://...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=clipzz-prod
PROCESS_VIDEO_ENDPOINT=https://...modal.run
PROCESS_VIDEO_ENDPOINT_AUTH=...
STRIPE_SECRET_KEY=sk_live_...
NEXT_PUBLIC_STRIPE_PUB_KEY=pk_live_...
STRIPE_SMALL_CREDIT_PACK=price_...
STRIPE_MEDIUM_CREDIT_PACK=price_...
STRIPE_LARGE_CREDIT_PACK=price_...
STRIPE_WEBHOOK_SECRET=whsec_...
RAZORPAY_KEY_ID=rzp_live_...
RAZORPAY_KEY_SECRET=...
RAZORPAY_WEBHOOK_SECRET=...
BASE_URL=https://clipzz.com
```

**<img src="https://fonts.gstatic.com/s/e/notoemoji/latest/26a0/512.gif" width="16"> Important:**
- Use **production** keys (not test keys)
- `BASE_URL` should be your production domain
- Rotate secrets from development

### 4. Deploy

Click **"Deploy"**

Vercel will:
1. Install dependencies
2. Run Prisma generate
3. Build Next.js app
4. Deploy to edge network

**Deployment URL:** `https://clipzz.vercel.app` (or custom domain)

### 5. Add Custom Domain

1. Go to **Project Settings → Domains**
2. Add `clipzz.com` and `www.clipzz.com`
3. Update DNS records:
   ```
   Type  Name  Value
   A     @     76.76.21.21
   CNAME www   cname.vercel-dns.com
   ```
4. Wait for DNS propagation (~15 minutes)

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4e6/512.gif" width="24"> Storage Deployment (AWS S3)

### 1. Create Production Bucket

```bash
# Create bucket
aws s3 mb s3://clipzz-prod --region us-east-1

# Enable versioning (optional)
aws s3api put-bucket-versioning \
  --bucket clipzz-prod \
  --versioning-configuration Status=Enabled
```

### 2. Configure CORS

```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST"],
    "AllowedOrigins": ["https://clipzz.com", "https://www.clipzz.com"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```

Apply:
```bash
aws s3api put-bucket-cors \
  --bucket clipzz-prod \
  --cors-configuration file://cors.json
```

### 3. Configure Lifecycle Rules (Cost Optimization)

```json
{
  "Rules": [
    {
      "Id": "DeleteOldVideos",
      "Status": "Enabled",
      "Filter": {},
      "Expiration": {
        "Days": 30
      }
    }
  ]
}
```

Apply:
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket clipzz-prod \
  --lifecycle-configuration file://lifecycle.json
```

### 4. Enable CloudFront (Optional)

For faster global delivery:

1. Go to **CloudFront** in AWS Console
2. Create distribution
3. Origin: `clipzz-prod.s3.amazonaws.com`
4. Enable caching
5. Update app to use CloudFront URLs

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4b3/512.gif" width="24"> Payment Gateway Configuration

### Stripe (Production)

1. **Switch to live mode** in Stripe Dashboard
2. Get **live API keys**:
   - `sk_live_...` (secret key)
   - `pk_live_...` (publishable key)
3. Create **webhook** for production:
   - URL: `https://clipzz.com/api/webhooks/stripe`
   - Events: `checkout.session.completed`
   - Get signing secret: `whsec_...`
4. Update environment variables in Vercel

### Razorpay (Production)

1. **Generate live keys** in Razorpay Dashboard
2. Get:
   - `rzp_live_...` (key ID)
   - Live key secret
3. Create **webhook**:
   - URL: `https://clipzz.com/api/webhooks/razorpay`
   - Event: `order.paid`
   - Get webhook secret
4. Update environment variables in Vercel

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f50d/512.gif" width="24"> Monitoring & Logging

### Application Monitoring

#### Option 1: Sentry (Recommended)

```bash
# Install
npm install @sentry/nextjs

# Configure
npx @sentry/wizard@latest -i nextjs

# Update sentry.client.config.ts
Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  tracesSampleRate: 0.1,
  environment: process.env.NODE_ENV,
});
```

#### Option 2: LogRocket

```bash
npm install logrocket
```

### Database Monitoring

**PostgreSQL:**
- Use provider's built-in monitoring (Vercel, Supabase, Railway)
- Set up alerts for:
  - Connection pool exhaustion
  - Slow queries
  - Disk usage

### S3 Monitoring

**CloudWatch:**
- Monitor storage size
- Track request counts
- Set billing alerts

```bash
# Enable S3 metrics
aws s3api put-bucket-metrics-configuration \
  --bucket clipzz-prod \
  --id AllRequests \
  --metrics-configuration Id=AllRequests,Filter={}
```

### Modal Monitoring

**Built-in Dashboard:**
- View function executions
- Monitor GPU usage
- Track costs
- View logs

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4be/512.gif" width="24"> Backup Strategy

### Database Backups

**Automated (Recommended):**
```bash
# Vercel Postgres: Automatic daily backups

# Manual backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d).sql

# Restore
psql $DATABASE_URL < backup_20250101.sql
```

**Backup retention:** 30 days minimum

### S3 Backups

**Cross-region replication:**
```bash
# Enable replication to us-west-2
aws s3api put-bucket-replication \
  --bucket clipzz-prod \
  --replication-configuration file://replication.json
```

**Versioning enabled:** Protects against accidental deletion

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f512/512.gif" width="24"> Security Hardening

### 1. Rotate Secrets

```bash
# Generate new AUTH_SECRET
openssl rand -base64 32

# Generate new AUTH_TOKEN
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Update in:
- Vercel environment variables
- Modal secrets

### 2. Enable Rate Limiting

Add to `next.config.js`:
```javascript
module.exports = {
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          {
            key: 'X-RateLimit-Limit',
            value: '100',
          },
        ],
      },
    ];
  },
};
```

### 3. Configure CSP Headers

```javascript
// next.config.js
const cspHeader = `
  default-src 'self';
  script-src 'self' 'unsafe-eval' 'unsafe-inline';
  style-src 'self' 'unsafe-inline';
  img-src 'self' blob: data:;
  font-src 'self';
  connect-src 'self' https://*.modal.run https://*.inngest.com;
  media-src 'self' https://*.s3.amazonaws.com;
`;
```

### 4. Enable HTTPS Everywhere

Vercel handles this automatically. Ensure:
- All API calls use HTTPS
- OAuth redirects use HTTPS
- Webhooks use HTTPS

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="24"> Post-Deployment Verification

### Smoke Tests

- [ ] Homepage loads
- [ ] Sign up works
- [ ] Sign in works
- [ ] Google OAuth works
- [ ] Upload video works
- [ ] Process video works
- [ ] View clips works
- [ ] Download clip works
- [ ] Buy credits works (Stripe)
- [ ] Buy credits works (Razorpay)
- [ ] Webhooks firing correctly

### Performance Tests

```bash
# Load test with k6
k6 run load-test.js

# Lighthouse audit
npx lighthouse https://clipzz.com --view
```

**Target metrics:**
- Time to First Byte (TTFB): <200ms
- First Contentful Paint (FCP): <1.5s
- Largest Contentful Paint (LCP): <2.5s
- Cumulative Layout Shift (CLS): <0.1

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f504/512.gif" width="24"> Rollback Strategy

### Vercel Rollback

```bash
# List deployments
vercel list

# Rollback to previous
vercel rollback [deployment-url]
```

Or use Vercel Dashboard:
1. Go to Deployments
2. Find previous working deployment
3. Click **"Promote to Production"**

### Modal Rollback

```bash
# Deploy previous version
git checkout <previous-commit>
modal deploy main.py

# Update frontend PROCESS_VIDEO_ENDPOINT if needed
```

### Database Rollback

```bash
# Restore from backup
psql $DATABASE_URL < backup_20250101.sql

# Or use provider's restore feature
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4ca/512.gif" width="24"> Cost Optimization

### Monthly Cost Estimate

| Service | Usage | Cost |
|---------|-------|------|
| Vercel | Hobby/Pro | $0-$20 |
| PostgreSQL | 1GB | $0-$15 |
| Modal | 100 videos/mo | ~$60 |
| S3 | 100GB + requests | ~$5 |
| Inngest | <1M steps | $0 |
| **Total** | | **~$70-$100/mo** |

### Optimization Tips

1. **Modal:** Use cheaper GPU (L4 vs L40S) for non-urgent jobs
2. **S3:** Enable lifecycle rules to delete old files
3. **Database:** Right-size connection pool
4. **CDN:** Cache static assets aggressively
5. **Inngest:** Batch operations where possible

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4de/512.gif" width="24"> Support Contacts

**Deployment Issues:**
- Vercel: [vercel.com/support](https://vercel.com/support)
- Modal: [modal.com/support](https://modal.com/support)
- AWS: [aws.amazon.com/support](https://aws.amazon.com/support)

**For questions:**
- GitHub Issues: [github.com/youruser/clipzz/issues](https://github.com/youruser/clipzz/issues)
- Email: devops@clipzz.com

---

<div align="center">
  <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f389/512.gif" width="32">
  <br><br>
  <strong>Deployment Complete!</strong>
  <br>
  Your app is now live in production.
</div>
