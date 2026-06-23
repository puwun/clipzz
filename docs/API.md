# <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4e1/512.gif" width="32"> API Documentation

Complete API reference for Clipzz backend endpoints and frontend server actions.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4cb/512.gif" width="24"> Table of Contents

- [Backend API (Modal)](#backend-api-modal)
- [Frontend Server Actions](#frontend-server-actions)
- [Webhook Endpoints](#webhook-endpoints)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f916/512.gif" width="24"> Backend API (Modal)

### Base URL
```
https://your-app--clipzz-video-processor-process-video.modal.run
```

Get your URL by running:
```bash
cd backend
modal deploy main.py
```

---

### POST /process_video

Process a video and generate vertical clips with subtitles.

**Authentication:** Bearer token (required)

**Headers:**
```http
Authorization: Bearer YOUR_AUTH_TOKEN
Content-Type: application/json
```

**Request Body:**
```json
{
  "s3_key": "userId/uploadId/original.mp4",
  "num_clips": 3
}
```

**Parameters:**

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `s3_key` | string | Yes | S3 object key of uploaded video | Must exist in S3 bucket |
| `num_clips` | integer | Yes | Number of clips to generate | Minimum: 1, Default: 3 |

**Response (Success):**
```json
{
  "status": "success",
  "clips_processed": 3
}
```

**Response (No Clips Found):**
```json
{
  "status": "success",
  "clips_processed": 0
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Error description here"
}
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success - clips processed |
| 401 | Unauthorized - invalid or missing bearer token |
| 422 | Validation error - invalid request body |
| 500 | Internal server error - processing failed |

**Example Request (cURL):**
```bash
curl -X POST "https://your-app.modal.run/process_video" \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_key": "clw123abc/01j456def/podcast.mp4",
    "num_clips": 3
  }'
```

**Example Request (JavaScript):**
```javascript
const response = await fetch(
  "https://your-app.modal.run/process_video",
  {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.PROCESS_VIDEO_ENDPOINT_AUTH}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      s3_key: "clw123abc/01j456def/podcast.mp4",
      num_clips: 3,
    }),
  }
);

const result = await response.json();
console.log(result);
// { status: "success", clips_processed: 3 }
```

**Processing Steps:**

1. **Download** video from S3
2. **Transcribe** audio with WhisperX (3-5 minutes)
3. **Analyze** transcript with Gemini AI (~10 seconds)
4. **For each clip:**
   - Extract segment
   - Detect active speaker (LR-ASD)
   - Create vertical video
   - Generate karaoke subtitles
   - Upload to S3
5. **Return** success response

**Typical Processing Time:**
- 1-hour podcast, 3 clips: ~12-15 minutes on L40S GPU

**Limitations:**
- Max timeout: 900 seconds (15 minutes)
- Longer videos may timeout - reduce `num_clips`
- No progress updates (fire-and-forget)

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/26a1/512.gif" width="24"> Frontend Server Actions

Server actions are type-safe, server-side functions callable from React components.

### Authentication Actions

#### `signUp(data)`

Create a new user account.

**Location:** `frontend/src/actions/auth.ts`

**Parameters:**
```typescript
{
  email: string;
  password: string;
  name: string;
}
```

**Returns:**
```typescript
Promise<{
  success: boolean;
  error?: string;
}>
```

**Example:**
```typescript
import { signUp } from "@/actions/auth";

const result = await signUp({
  email: "user@example.com",
  password: "securePass123",
  name: "John Doe",
});

if (result.success) {
  // Redirect to dashboard
} else {
  console.error(result.error);
}
```

**Side Effects:**
- Creates User record with 10 initial credits
- Hashes password with bcrypt (12 rounds)
- Creates Razorpay contact (if available)

---

### S3 Actions

#### `generateUploadUrl(fileInfo)`

Generate presigned URL for direct S3 upload.

**Location:** `frontend/src/actions/s3.ts`

**Parameters:**
```typescript
{
  filename: string;
  fileType: string;  // e.g., "video/mp4"
  fileSize: number;  // bytes
}
```

**Returns:**
```typescript
Promise<{
  uploadUrl: string;      // Presigned PUT URL
  s3Key: string;          // Object key in S3
  uploadedFileId: string; // Database record ID
}>
```

**Example:**
```typescript
import { generateUploadUrl } from "@/actions/s3";

const file = document.querySelector("input[type=file]").files[0];

const { uploadUrl, s3Key, uploadedFileId } = await generateUploadUrl({
  filename: file.name,
  fileType: file.type,
  fileSize: file.size,
});

// Upload directly to S3
await fetch(uploadUrl, {
  method: "PUT",
  body: file,
  headers: { "Content-Type": file.type },
});

// Now process the video
await processVideo(uploadedFileId, 3);
```

**URL Expiration:** 600 seconds (10 minutes)

**S3 Key Format:** `{userId}/{uuid}/{filename}`

---

### Video Processing Actions

#### `processVideo(uploadedFileId, numClips)`

Trigger background job to process uploaded video.

**Location:** `frontend/src/actions/generate.ts`

**Parameters:**
```typescript
uploadedFileId: string;  // From generateUploadUrl
numClips: number;        // 1+
```

**Returns:**
```typescript
Promise<{
  success: boolean;
  error?: string;
}>
```

**Example:**
```typescript
import { processVideo } from "@/actions/generate";

const result = await processVideo(uploadedFileId, 3);

if (result.success) {
  console.log("Processing started!");
  // Poll database for status updates
}
```

**Side Effects:**
- Sets `uploaded: true` on UploadedFile
- Sends `video/process` event to Inngest
- Inngest job checks credits and calls Modal

**Processing Flow:**
```
processVideo()
  → Inngest event
  → Credit check
  → Modal API call
  → Video processing
  → Clips uploaded to S3
  → Database updated
```

---

#### `getClipPlayUrl(clipId)`

Get presigned URL to view/download a clip.

**Location:** `frontend/src/actions/generate.ts`

**Parameters:**
```typescript
clipId: string;  // Clip database ID
```

**Returns:**
```typescript
Promise<{
  url: string;   // Presigned GET URL
  error?: string;
}>
```

**Example:**
```typescript
import { getClipPlayUrl } from "@/actions/generate";

const { url } = await getClipPlayUrl("clip_abc123");

// Use in video player
<video src={url} controls />
```

**URL Expiration:** 3600 seconds (1 hour)

**Authorization:** Only clip owner can generate URL

---

### Payment Actions

#### `createCheckoutSession(priceId)` (Stripe)

Create Stripe checkout session for credit purchase.

**Location:** `frontend/src/actions/stripe.ts`

**Parameters:**
```typescript
priceId: string;  // Stripe price ID (STRIPE_SMALL_CREDIT_PACK, etc.)
```

**Returns:**
```typescript
Promise<{
  url: string;  // Stripe checkout URL
}>
```

**Example:**
```typescript
import { createCheckoutSession } from "@/actions/stripe";

const { url } = await createCheckoutSession(
  process.env.STRIPE_SMALL_CREDIT_PACK
);

// Redirect to Stripe
window.location.href = url;
```

**Credit Packs:**

| Price ID | Credits | Price |
|----------|---------|-------|
| `STRIPE_SMALL_CREDIT_PACK` | 50 | $10 |
| `STRIPE_MEDIUM_CREDIT_PACK` | 150 | $25 |
| `STRIPE_LARGE_CREDIT_PACK` | 500 | $70 |

**Success Flow:**
1. User redirected to Stripe
2. Completes payment
3. Stripe sends webhook to `/api/webhooks/stripe`
4. Credits added to user account
5. User redirected back to app

---

#### `createCheckoutSession(priceId)` (Razorpay)

Create Razorpay checkout for credit purchase (India).

**Location:** `frontend/src/actions/razorpay.ts`

**Parameters:**
```typescript
priceId: "small" | "medium" | "large";
```

**Returns:**
```typescript
Promise<{
  orderId: string;
  amount: number;
  currency: "INR";
}>
```

**Example:**
```typescript
import { createCheckoutSession } from "@/actions/razorpay";

const { orderId, amount } = await createCheckoutSession("small");

// Open Razorpay modal
const razorpay = new Razorpay({
  key: process.env.NEXT_PUBLIC_RAZORPAY_KEY_ID,
  order_id: orderId,
  amount: amount,
  handler: (response) => {
    console.log("Payment successful:", response);
  },
});

razorpay.open();
```

**Credit Packs:**

| Pack ID | Credits | Price (INR) |
|---------|---------|-------------|
| `small` | 50 | ₹830 |
| `medium` | 150 | ₹2,075 |
| `large` | 500 | ₹5,810 |

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f514/512.gif" width="24"> Webhook Endpoints

### POST /api/webhooks/stripe

Handles Stripe payment events.

**Location:** `frontend/src/app/api/webhooks/stripe/route.ts`

**Event:** `checkout.session.completed`

**Signature Verification:** Required (using `STRIPE_WEBHOOK_SECRET`)

**Request Headers:**
```http
Stripe-Signature: t=timestamp,v1=signature
```

**Request Body:**
```json
{
  "type": "checkout.session.completed",
  "data": {
    "object": {
      "id": "cs_test_...",
      "customer_email": "user@example.com",
      "amount_total": 1000,
      "metadata": {
        "priceId": "price_1234..."
      }
    }
  }
}
```

**Credit Allocation:**
- $10 (1000 cents) → 50 credits
- $25 (2500 cents) → 150 credits
- $70 (7000 cents) → 500 credits

**Response:**
```json
{ "received": true }
```

**Testing Locally:**
```bash
# Install Stripe CLI
stripe listen --forward-to localhost:3000/api/webhooks/stripe

# Use test webhook secret in .env
STRIPE_WEBHOOK_SECRET=whsec_...
```

---

### POST /api/webhooks/razorpay

Handles Razorpay payment events.

**Location:** `frontend/src/app/api/webhooks/razorpay/route.ts`

**Event:** `order.paid`

**Signature Verification:** Required (using `RAZORPAY_WEBHOOK_SECRET`)

**Request Headers:**
```http
X-Razorpay-Signature: signature_here
```

**Request Body:**
```json
{
  "event": "order.paid",
  "payload": {
    "order": {
      "entity": {
        "id": "order_...",
        "amount": 83000,
        "currency": "INR"
      }
    },
    "payment": {
      "entity": {
        "email": "user@example.com"
      }
    }
  }
}
```

**Credit Allocation:**
- ₹830 → 50 credits
- ₹2,075 → 150 credits
- ₹5,810 → 500 credits

**Response:**
```json
{ "status": "ok" }
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f6a8/512.gif" width="24"> Error Handling

### Common Error Responses

#### 401 Unauthorized
```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing authentication token"
}
```

**Causes:**
- Missing `Authorization` header
- Invalid bearer token
- Expired session

**Solution:** Re-authenticate user

---

#### 403 Forbidden
```json
{
  "error": "Forbidden",
  "message": "You don't have permission to access this resource"
}
```

**Causes:**
- Accessing another user's clip
- Insufficient credits

**Solution:** Check authorization, verify user owns resource

---

#### 422 Validation Error
```json
{
  "error": "Validation failed",
  "details": [
    {
      "field": "num_clips",
      "message": "Must be at least 1"
    }
  ]
}
```

**Causes:**
- Invalid request parameters
- Missing required fields

**Solution:** Validate input on client-side

---

#### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "An unexpected error occurred"
}
```

**Causes:**
- Modal processing failure
- Database connection issues
- S3 upload/download errors

**Solution:** Retry request, check logs

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/23f1/512.gif" width="24"> Rate Limits

### Current Limits (Inngest)

**Video Processing:**
- **Concurrency:** 1 job per user
- **Retry:** Single retry on failure
- **Timeout:** 900 seconds (15 minutes)

**Example:** If a user starts processing while another job is running, the second job queues until the first completes.

### Recommended Production Limits

**API Endpoints:**
```typescript
// Implement rate limiting
import rateLimit from "express-rate-limit";

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Max 100 requests per window
});
```

**Upload URLs:**
- 10 per minute per user

**Processing:**
- 5 active jobs per user
- 100 jobs per day per user

