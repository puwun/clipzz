# <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f512/512.gif" width="32"> Security Documentation

Security best practices and vulnerability management for Clipzz.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/26a0/512.gif" width="24"> Security Overview

Clipzz implements multiple security layers to protect user data and prevent unauthorized access:

- **Authentication:** NextAuth.js with JWT sessions
- **Authorization:** Server-side permission checks
- **Data Encryption:** HTTPS/TLS in transit, encrypted at rest (S3)
- **Payment Security:** PCI-compliant gateways (Stripe/Razorpay)
- **Input Validation:** Zod schemas on all endpoints
- **SQL Injection Prevention:** Prisma ORM with parameterized queries

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f510/512.gif" width="24"> Authentication & Authorization

### NextAuth.js Configuration

**Session Strategy:** JWT (stateless)

```typescript
// src/lib/auth.ts
export const authConfig = {
  session: {
    strategy: "jwt",
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },
  callbacks: {
    jwt({ token, user }) {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
    session({ session, token }) {
      session.user.id = token.id;
      return session;
    },
  },
};
```

**Security features:**
- HTTP-only cookies
- Secure flag in production
- SameSite: Lax
- CSRF protection built-in

### Password Security

**Hashing:** bcrypt with 12 rounds

```typescript
// src/lib/auth.ts
export async function hashPassword(password: string) {
  return bcrypt.hash(password, 12); // 12 rounds = ~250ms
}

export async function verifyPassword(password: string, hash: string) {
  return bcrypt.compare(password, hash);
}
```

**Requirements:**
- Minimum 8 characters (enforced client-side)
- Should enforce: 1 uppercase, 1 lowercase, 1 number (recommended)

### Authorization Checks

**Example: Clip access control**

```typescript
// src/actions/generate.ts
export async function getClipPlayUrl(clipId: string) {
  const session = await auth();
  if (!session?.user?.id) {
    throw new Error("Unauthorized");
  }

  const clip = await prisma.clip.findUnique({
    where: { id: clipId },
  });

  if (!clip || clip.userId !== session.user.id) {
    throw new Error("Forbidden"); // Not your clip!
  }

  // Generate presigned URL
  ...
}
```

**Best practice:** Always verify user owns resource before access.

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f511/512.gif" width="24"> API Security

### Bearer Token Authentication (Modal)

```python
# backend/main.py
def process_video(request: ProcessVideoRequest, token: HTTPAuthorizationCredentials):
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
        )
```

**Token Management:**
- Store in environment variables
- Rotate every 90 days
- Use strong random tokens (32+ characters)

**Generate secure token:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Input Validation

**Frontend validation (Zod):**

```typescript
// src/actions/generate.ts
const schema = z.object({
  uploadedFileId: z.string().uuid(),
  numClips: z.number().min(1).max(20),
});

export async function processVideo(uploadedFileId: string, numClips: number) {
  const validated = schema.parse({ uploadedFileId, numClips });
  // ... process
}
```

**Backend validation (Pydantic):**

```python
# backend/main.py
class ProcessVideoRequest(BaseModel):
    s3_key: str
    num_clips: int = Field(ge=1, description="Must be at least 1")
```

**<img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="16"> Best practice:** Validate on both client and server.

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4b3/512.gif" width="24"> Payment Security

### Webhook Signature Verification

#### Stripe

```typescript
// src/app/api/webhooks/stripe/route.ts
const signature = headers().get("stripe-signature");
const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET!;

try {
  const event = stripe.webhooks.constructEvent(
    body,
    signature,
    webhookSecret
  );
  // Process event
} catch (err) {
  return Response.json({ error: "Invalid signature" }, { status: 400 });
}
```

#### Razorpay

```typescript
// src/app/api/webhooks/razorpay/route.ts
const signature = headers().get("x-razorpay-signature");
const secret = process.env.RAZORPAY_WEBHOOK_SECRET!;

const expectedSignature = crypto
  .createHmac("sha256", secret)
  .update(body)
  .digest("hex");

if (signature !== expectedSignature) {
  return Response.json({ error: "Invalid signature" }, { status: 400 });
}
```

**<img src="https://fonts.gstatic.com/s/e/notoemoji/latest/26a0/512.gif" width="16"> Critical:** Never process webhooks without signature verification!

### PCI Compliance

**Stripe/Razorpay handle:**
- Credit card data (never touches your server)
- PCI DSS compliance
- 3D Secure authentication
- Fraud detection

**Your responsibility:**
- Secure webhook endpoints (HTTPS only)
- Verify signatures
- Store transaction IDs only (not card details)

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4be/512.gif" width="24"> Data Security

### Encryption at Rest

**S3 Storage:**
```bash
# Enable default encryption
aws s3api put-bucket-encryption \
  --bucket clipzz-prod \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

**Database:**
- PostgreSQL: Enable encryption at rest (provider-dependent)
- Prisma: Connections over SSL/TLS

### Encryption in Transit

**HTTPS Everywhere:**
- Vercel: Automatic HTTPS (Let's Encrypt)
- Modal: HTTPS endpoints
- S3: Use presigned URLs with HTTPS only

**Enforce HTTPS:**
```typescript
// next.config.js
module.exports = {
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload',
          },
        ],
      },
    ];
  },
};
```

### Presigned URL Security

**S3 URLs:**
- Upload: 600 seconds expiry (10 minutes)
- Download: 3600 seconds expiry (1 hour)
- Generated per-request (not cached)

```typescript
// src/actions/s3.ts
const command = new PutObjectCommand({
  Bucket: process.env.S3_BUCKET_NAME!,
  Key: s3Key,
});

const uploadUrl = await getSignedUrl(s3Client, command, {
  expiresIn: 600, // 10 minutes
});
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f6ab/512.gif" width="24"> Common Vulnerabilities

### SQL Injection <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="16"> Protected

**Risk:** Attackers inject SQL via user input

**Protection:** Prisma uses parameterized queries

```typescript
// Good: Prisma (safe)
const user = await prisma.user.findUnique({
  where: { email: userInput }, // Automatically escaped
});

// Bad: Raw SQL (vulnerable)
const user = await prisma.$queryRaw`
  SELECT * FROM User WHERE email = '${userInput}'
`; // Don't do this!
```

### XSS (Cross-Site Scripting) <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="16"> Protected

**Risk:** Inject malicious scripts via user input

**Protection:** React escapes by default

```tsx
// Good: React (safe)
<div>{userInput}</div> // Automatically escaped

// Bad: dangerouslySetInnerHTML (vulnerable)
<div dangerouslySetInnerHTML={{ __html: userInput }} />
// Only use if you sanitize first!
```

**When using HTML:**
```typescript
import DOMPurify from 'dompurify';

const clean = DOMPurify.sanitize(userInput);
<div dangerouslySetInnerHTML={{ __html: clean }} />
```

### CSRF (Cross-Site Request Forgery) <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="16"> Protected

**Risk:** Unauthorized actions via forged requests

**Protection:** NextAuth includes CSRF tokens

```typescript
// Built-in: NextAuth handles CSRF
// Session cookies have SameSite: Lax
// State parameter in OAuth flows
```

### Command Injection <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/26a0/512.gif" width="16"> Partial

**Risk:** Execute shell commands via user input

**Current code:**
```python
# backend/main.py (SAFE - no user input in command)
ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                  f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                  f"{output_path}")
subprocess.run(ffmpeg_command, shell=True, check=True)
```

**<img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="16"> Safe because:** Paths are internally generated, not from user input.

**If using user input:**
```python
# Bad: Never do this
filename = request.filename  # User input!
subprocess.run(f"ffmpeg -i {filename} output.mp4", shell=True)

# Good: Use shlex.quote()
import shlex
filename = shlex.quote(request.filename)
subprocess.run(f"ffmpeg -i {filename} output.mp4", shell=True)

# Better: Use list (no shell)
subprocess.run(["ffmpeg", "-i", request.filename, "output.mp4"])
```

### Insecure Direct Object References (IDOR) <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="16"> Protected

**Risk:** Access resources by guessing IDs

**Protection:** Always check ownership

```typescript
// Good: Authorization check
export async function getClip(clipId: string) {
  const session = await auth();
  const clip = await prisma.clip.findUnique({
    where: { id: clipId },
  });

  if (clip.userId !== session.user.id) {
    throw new Error("Forbidden");
  }

  return clip;
}

// Bad: No authorization
export async function getClip(clipId: string) {
  return prisma.clip.findUnique({
    where: { id: clipId },
  }); // Anyone can access any clip!
}
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4a1/512.gif" width="24"> Security Best Practices

### 1. Environment Variables

**Never commit:**
```bash
# .gitignore (already included)
.env
.env.local
.env.*.local
```

**Verify secrets not committed:**
```bash
git log -p | grep -i "api_key\|secret\|password"
```

**Rotate regularly:**
- `AUTH_SECRET`: Every 90 days
- `AUTH_TOKEN`: Every 90 days
- API keys: On security incidents

### 2. Dependency Updates

**Check for vulnerabilities:**
```bash
# Frontend
npm audit
npm audit fix

# Backend
pip install safety
safety check
```

**Automate with Dependabot:**
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/frontend"
    schedule:
      interval: "weekly"

  - package-ecosystem: "pip"
    directory: "/backend"
    schedule:
      interval: "weekly"
```

### 3. Rate Limiting

**Implement on critical endpoints:**

```typescript
// middleware.ts (example with upstash/ratelimit)
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(10, "10 s"),
});

export async function middleware(request: NextRequest) {
  const ip = request.ip ?? "127.0.0.1";
  const { success } = await ratelimit.limit(ip);

  if (!success) {
    return new Response("Too Many Requests", { status: 429 });
  }

  return NextResponse.next();
}
```

### 4. Content Security Policy

```typescript
// next.config.js
const cspHeader = `
  default-src 'self';
  script-src 'self' 'unsafe-eval' 'unsafe-inline' https://js.stripe.com;
  style-src 'self' 'unsafe-inline';
  img-src 'self' blob: data: https:;
  font-src 'self';
  connect-src 'self' https://*.modal.run https://*.inngest.com;
  media-src 'self' https://*.s3.amazonaws.com;
  frame-src https://js.stripe.com;
`;

module.exports = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: cspHeader.replace(/\n/g, ''),
          },
        ],
      },
    ];
  },
};
```

### 5. Secure Headers

```typescript
// next.config.js
async headers() {
  return [
    {
      source: '/:path*',
      headers: [
        {
          key: 'X-DNS-Prefetch-Control',
          value: 'on',
        },
        {
          key: 'Strict-Transport-Security',
          value: 'max-age=63072000; includeSubDomains; preload',
        },
        {
          key: 'X-Frame-Options',
          value: 'SAMEORIGIN',
        },
        {
          key: 'X-Content-Type-Options',
          value: 'nosniff',
        },
        {
          key: 'Referrer-Policy',
          value: 'origin-when-cross-origin',
        },
      ],
    },
  ];
}
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f6a8/512.gif" width="24"> Incident Response

### Reporting a Vulnerability

**Please report security issues privately:**

**Email:** security@clipzz.com

**Include:**
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

**What NOT to do:**
- Post publicly before we've patched
- Exploit the vulnerability
- Test on production data

**Response time:**
- Acknowledgment: 24 hours
- Initial assessment: 72 hours
- Fix timeline: Depends on severity

### Security Incident Playbook

**If breach detected:**

1. **Contain**
   - Revoke compromised credentials immediately
   - Block malicious IPs
   - Take affected services offline if needed

2. **Assess**
   - Identify scope (what data accessed?)
   - Review logs (Modal, Vercel, S3, DB)
   - Document timeline

3. **Notify**
   - Affected users (if PII exposed)
   - Payment processors (if payment data involved)
   - Regulators (GDPR compliance if EU users)

4. **Remediate**
   - Patch vulnerability
   - Rotate all secrets
   - Deploy fix

5. **Post-mortem**
   - Document incident
   - Update security practices
   - Conduct team training

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/2705/512.gif" width="24"> Security Checklist

**Before deploying to production:**

- [ ] All secrets rotated from development
- [ ] HTTPS enforced everywhere
- [ ] Webhook signatures verified
- [ ] Authorization checks on all endpoints
- [ ] Input validation with Zod/Pydantic
- [ ] SQL injection prevented (using ORM)
- [ ] XSS prevented (React auto-escaping)
- [ ] CSRF protection enabled (NextAuth)
- [ ] Rate limiting implemented
- [ ] Dependencies up to date (no critical CVEs)
- [ ] S3 bucket encryption enabled
- [ ] S3 CORS configured correctly
- [ ] Database over SSL/TLS
- [ ] Security headers configured
- [ ] CSP policy defined
- [ ] Error messages don't leak info
- [ ] Logging doesn't include secrets
- [ ] Backup and disaster recovery plan
- [ ] Incident response plan documented

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4da/512.gif" width="24"> Security Resources

### Tools

- **Snyk:** Dependency vulnerability scanning
- **OWASP ZAP:** Penetration testing
- **Burp Suite:** Web security testing
- **npm audit:** Node.js security
- **Safety:** Python security

### References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NextAuth.js Security](https://next-auth.js.org/configuration/options#security)
- [Stripe Security](https://stripe.com/docs/security)
- [AWS S3 Security](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security.html)

---

<div align="center">
  <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f512/512.gif" width="32">
  <br><br>
  <strong>Security is a continuous process, not a one-time task.</strong>
  <br>
  Stay vigilant and keep systems updated.
</div>
