# <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f5c3/512.gif" width="32"> Database Schema Documentation

Complete documentation of Clipzz's database schema using Prisma ORM.

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4ca/512.gif" width="24"> Overview

Clipzz uses **Prisma ORM** for type-safe database access with:
- **SQLite** for development (file-based, no setup required)
- **PostgreSQL** recommended for production (scalable, concurrent access)

**Schema Location:** [`frontend/prisma/schema.prisma`](frontend/prisma/schema.prisma)

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f5fa/512.gif" width="24"> Entity Relationship Diagram

```
┌──────────────────────┐
│       User           │
│──────────────────────│
│ id (PK)              │
│ email (unique)       │
│ name                 │
│ password             │
│ credits (default 10) │────┐
│ razorpayContactId    │    │
└──────┬───────────────┘    │
       │                    │
       │ 1:N                │ 1:N
       │                    │
       ↓                    ↓
┌──────────────────────┐  ┌────────────────────┐
│   UploadedFile       │  │      Clip          │
│──────────────────────│  │────────────────────│
│ id (PK)              │  │ id (PK)            │
│ s3Key                │←─┤ uploadedFileId (FK)│
│ displayName          │  │ s3Key              │
│ uploaded (bool)      │  │ userId (FK)        │
│ status               │  │ createdAt          │
│ numClips             │  └────────────────────┘
│ userId (FK)          │
│ createdAt            │
└──────────────────────┘

┌──────────────────────┐  ┌────────────────────┐
│      Account         │  │     Session        │
│ (NextAuth OAuth)     │  │  (NextAuth JWT)    │
│──────────────────────│  │────────────────────│
│ id (PK)              │  │ id (PK)            │
│ userId (FK) ─────────┼──┤ userId (FK)        │
│ provider (google)    │  │ sessionToken       │
│ providerAccountId    │  │ expires            │
│ access_token         │  └────────────────────┘
│ refresh_token        │
└──────────────────────┘

┌──────────────────────┐  ┌────────────────────┐
│   Post (unused)      │  │ VerificationToken  │
│──────────────────────│  │  (NextAuth email)  │
│ id (PK)              │  │────────────────────│
│ name                 │  │ identifier         │
│ createdById (FK)     │  │ token              │
└──────────────────────┘  │ expires            │
                          └────────────────────┘
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f465/512.gif" width="24"> Core Models

### User

Central model for authentication and credit management.

```prisma
model User {
  id                String    @id @default(cuid())
  name              String?
  email             String    @unique
  emailVerified     DateTime?
  password          String?
  credits           Int       @default(10)
  razorpayContactId String?   @unique
  image             String?
  accounts          Account[]
  sessions          Session[]
  posts             Post[]
  uploadedFiles     UploadedFile[]
  clips             Clip[]
}
```

#### Field Descriptions

| Field | Type | Description | Default | Required |
|-------|------|-------------|---------|----------|
| `id` | String (CUID) | Unique user identifier | Auto-generated | Yes |
| `email` | String | User's email address (unique) | - | Yes |
| `name` | String? | Display name | `null` | No |
| `password` | String? | Hashed password (bcrypt, 12 rounds) | `null` | No* |
| `credits` | Int | Available credits for clip generation | 10 | Yes |
| `razorpayContactId` | String? | Razorpay customer ID for payments | `null` | No |
| `emailVerified` | DateTime? | Email verification timestamp | `null` | No |
| `image` | String? | Profile image URL (from OAuth) | `null` | No |

**\* Note:** `password` is null for OAuth users (Google sign-in), required for credentials users.

#### Relationships

- **1:N with UploadedFile** - User can upload multiple videos
- **1:N with Clip** - User owns multiple clips
- **1:N with Account** - OAuth accounts (Google, etc.)
- **1:N with Session** - Active sessions
- **1:N with Post** - Legacy model (unused)

#### Credit System

- **Initial Credits:** 10 (free trial)
- **Deduction:** 1 credit = 1 clip generated
- **Top-Up:** Purchase credit packs via Stripe/Razorpay
- **Never Expire:** Credits remain until used

#### Business Rules

```typescript
// Check if user has enough credits
if (user.credits >= numClips) {
  // Process video
  // Deduct credits AFTER processing completes
  await prisma.user.update({
    where: { id: userId },
    data: { credits: { decrement: numClips } }
  });
}
```

---

### UploadedFile

Tracks video uploads and processing status.

```prisma
model UploadedFile {
  id            String   @id @default(uuid())
  s3Key         String
  displayName   String?
  uploaded      Boolean  @default(false)
  status        String   @default("queued")
  numClips      Int?
  createdAt     DateTime @default(now())
  updatedAt     DateTime @updatedAt

  clips         Clip[]
  user          User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  userId        String

  @@index([s3Key])
}
```

#### Field Descriptions

| Field | Type | Description | Default | Notes |
|-------|------|-------------|---------|-------|
| `id` | UUID | Unique file identifier | Auto-generated | Primary key |
| `s3Key` | String | S3 object key: `{userId}/{uuid}/{filename}` | - | Indexed |
| `displayName` | String? | Original filename | `null` | Optional |
| `uploaded` | Boolean | Whether file has been uploaded to S3 | `false` | Prevents duplicate processing |
| `status` | String | Processing status (see below) | `"queued"` | State machine |
| `numClips` | Int? | Number of clips requested | `null` | Set when processing starts |
| `createdAt` | DateTime | Record creation timestamp | `now()` | Auto |
| `updatedAt` | DateTime | Last update timestamp | `now()` | Auto-updated |

#### Status State Machine

```
┌─────────┐
│ queued  │ Initial state (user uploaded video)
└────┬────┘
     │
     ↓
┌────────────┐
│ processing │ Inngest job running on Modal
└─────┬──────┘
      │
      ├──→ processed     ✓ Success (clips created)
      ├──→ no-credits    ✗ Insufficient credits
      └──→ failed        ✗ Processing error
```

**Valid Status Values:**
- `"queued"` - Waiting to be processed
- `"processing"` - Currently being processed by Modal
- `"processed"` - Successfully processed, clips available
- `"no-credits"` - User doesn't have enough credits
- `"failed"` - Processing encountered an error

#### Uploaded Flag

**Purpose:** Prevents reprocessing the same video multiple times.

```typescript
// Check before sending to Inngest
if (uploadedFile.uploaded) {
  throw new Error("This video has already been processed");
}

// Set to true when Inngest event triggered
await prisma.uploadedFile.update({
  where: { id: uploadedFileId },
  data: { uploaded: true }
});
```

#### Relationships

- **N:1 with User** - Each file belongs to one user
- **1:N with Clip** - One video generates multiple clips

---

### Clip

Stores metadata for generated clips.

```prisma
model Clip {
  id             String   @id @default(uuid())
  s3Key          String
  createdAt      DateTime @default(now())
  updatedAt      DateTime @updatedAt

  uploadedFile   UploadedFile? @relation(fields: [uploadedFileId], references: [id], onDelete: Cascade)
  uploadedFileId String?
  user           User          @relation(fields: [userId], references: [id], onDelete: Cascade)
  userId         String

  @@index([s3Key])
}
```

#### Field Descriptions

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| `id` | UUID | Unique clip identifier | Primary key |
| `s3Key` | String | S3 object key: `{userId}/{uploadId}/clip_{N}.mp4` | Indexed for fast lookups |
| `uploadedFileId` | UUID? | Reference to source video | Optional (can be orphaned) |
| `userId` | String | Owner of the clip | Required |
| `createdAt` | DateTime | Clip creation timestamp | Auto |
| `updatedAt` | DateTime | Last update timestamp | Auto |

#### S3 Key Format

```
{userId}/{uploadId}/clip_0.mp4
{userId}/{uploadId}/clip_1.mp4
{userId}/{uploadId}/clip_2.mp4
```

**Example:**
```
clw123abc/01j123def/clip_0.mp4
clw123abc/01j123def/clip_1.mp4
```

#### Cascade Deletion

When a User or UploadedFile is deleted, all associated Clips are automatically deleted (`onDelete: Cascade`).

```typescript
// Deleting user also deletes all their clips
await prisma.user.delete({ where: { id: userId } });
// → All Clip records with userId deleted
// → S3 files remain (requires manual cleanup)
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f510/512.gif" width="24"> Authentication Models (NextAuth.js)

### Account

Stores OAuth provider credentials.

```prisma
model Account {
  id                       String  @id @default(cuid())
  userId                   String
  type                     String
  provider                 String
  providerAccountId        String
  refresh_token            String?
  access_token             String?
  expires_at               Int?
  token_type               String?
  scope                    String?
  id_token                 String?
  session_state            String?
  refresh_token_expires_in Int?
  user                     User    @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([provider, providerAccountId])
}
```

**Purpose:** Links User to OAuth providers (Google, etc.)

**Example Record:**
```json
{
  "id": "clw456xyz",
  "userId": "clw123abc",
  "type": "oauth",
  "provider": "google",
  "providerAccountId": "1234567890",
  "access_token": "ya29.a0...",
  "refresh_token": "1//0g...",
  "expires_at": 1699999999,
  "token_type": "Bearer",
  "scope": "openid email profile"
}
```

### Session

Stores active user sessions (JWT-based).

```prisma
model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  expires      DateTime
  user         User     @relation(fields: [userId], references: [id], onDelete: Cascade)
}
```

**Purpose:** Tracks active user sessions for authentication.

### VerificationToken

Email verification tokens (email/password flow).

```prisma
model VerificationToken {
  identifier String
  token      String   @unique
  expires    DateTime

  @@unique([identifier, token])
}
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/274c/512.gif" width="24"> Legacy Models

### Post

**Status:** UNUSED - Can be removed

```prisma
model Post {
  id          Int      @id @default(autoincrement())
  name        String
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
  createdBy   User     @relation(fields: [createdById], references: [id])
  createdById String

  @@index([name])
}
```

This model is from the T3 Stack boilerplate and is not used in Clipzz.

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f50d/512.gif" width="24"> Indexes

Indexes improve query performance:

| Model | Field | Purpose |
|-------|-------|---------|
| UploadedFile | `s3Key` | Fast lookup by S3 key |
| Clip | `s3Key` | Fast lookup by S3 key |
| Post | `name` | Legacy (unused) |

**Missing Indexes (Recommended):**
```prisma
// Add these for better performance
model User {
  @@index([email])  // Fast lookup by email (login)
}

model UploadedFile {
  @@index([userId, status])  // Fast queries: "user's processed files"
  @@index([createdAt])       // Sort by upload time
}

model Clip {
  @@index([userId])          // Fast queries: "user's clips"
  @@index([uploadedFileId])  // Fast queries: "clips from this video"
}
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f6e0/512.gif" width="24"> Database Operations

### Initial Setup

```bash
# Install Prisma CLI
cd frontend
npm install

# Generate Prisma Client
npx prisma generate

# Create database and tables (SQLite)
npx prisma db push

# Or run migrations (PostgreSQL)
npx prisma migrate deploy
```

### Common Queries

#### Create User with Initial Credits
```typescript
const user = await prisma.user.create({
  data: {
    email: "user@example.com",
    name: "John Doe",
    password: hashedPassword,
    credits: 10,  // Default free credits
  }
});
```

#### Create Upload Record
```typescript
const upload = await prisma.uploadedFile.create({
  data: {
    s3Key: `${userId}/${uploadId}/video.mp4`,
    displayName: "my-podcast.mp4",
    userId: userId,
    status: "queued",
  }
});
```

#### Update Processing Status
```typescript
await prisma.uploadedFile.update({
  where: { id: uploadId },
  data: {
    status: "processing",
    numClips: 3,
  }
});
```

#### Create Clips After Processing
```typescript
await prisma.clip.createMany({
  data: [
    {
      s3Key: `${userId}/${uploadId}/clip_0.mp4`,
      userId: userId,
      uploadedFileId: uploadId,
    },
    {
      s3Key: `${userId}/${uploadId}/clip_1.mp4`,
      userId: userId,
      uploadedFileId: uploadId,
    },
  ]
});
```

#### Deduct Credits
```typescript
await prisma.user.update({
  where: { id: userId },
  data: {
    credits: { decrement: numClips }  // Atomic operation
  }
});
```

#### Query User's Uploads with Clips
```typescript
const uploads = await prisma.uploadedFile.findMany({
  where: { userId: userId },
  include: {
    clips: true,
  },
  orderBy: { createdAt: 'desc' },
});
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4c8/512.gif" width="24"> Migrations

### Creating Migrations (PostgreSQL)

```bash
# Create a new migration
npx prisma migrate dev --name add_feature

# Apply migrations in production
npx prisma migrate deploy

# Reset database (DEV ONLY - deletes all data)
npx prisma migrate reset
```

### SQLite to PostgreSQL Migration

1. **Update `schema.prisma`:**
```prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

// Uncomment @db.Text annotations in Account model
model Account {
  refresh_token String? @db.Text
  access_token  String? @db.Text
  id_token      String? @db.Text
}
```

2. **Update DATABASE_URL:**
```bash
DATABASE_URL="postgresql://user:password@host:5432/clipzz?schema=public"
```

3. **Create initial migration:**
```bash
npx prisma migrate dev --name init
```

4. **Export/Import data** (if needed):
```bash
# Export from SQLite
sqlite3 dev.db .dump > data.sql

# Convert and import to PostgreSQL
# (requires manual conversion of SQLite → PostgreSQL syntax)
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/26a0/512.gif" width="24"> Best Practices

### 1. Use Transactions for Credit Operations

```typescript
// Bad: Race condition possible
const user = await prisma.user.findUnique({ where: { id } });
if (user.credits >= numClips) {
  await processVideo();  // ← Another request could deduct credits here
  await prisma.user.update({ data: { credits: { decrement: numClips } } });
}

// Good: Atomic operation
await prisma.user.update({
  where: {
    id: userId,
    credits: { gte: numClips }  // ✓ Check AND update atomically
  },
  data: { credits: { decrement: numClips } }
});
```

### 2. Use Cascade Deletion

Already configured: Deleting a User or UploadedFile automatically deletes related records.

### 3. Handle S3 Cleanup Separately

Database cascades don't delete S3 files. Implement separate cleanup:

```typescript
// Delete from database
await prisma.uploadedFile.delete({ where: { id } });

// Separately delete from S3
await s3.deleteObject({
  Bucket: "clipzz",
  Key: file.s3Key,
});
```

### 4. Connection Pooling (PostgreSQL)

```bash
# .env
DATABASE_URL="postgresql://user:pass@host:5432/clipzz?schema=public&connection_limit=5&pool_timeout=10"
```

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f4a1/512.gif" width="24"> Schema Evolution

### Current Version
- SQLite for development
- 7 models (5 active, 2 unused/legacy)
- Basic indexes on S3 keys

### Recommended Improvements
- [ ] Migrate to PostgreSQL for production
- [ ] Add composite indexes for common queries
- [ ] Remove unused `Post` model
- [ ] Add `Clip.clipIndex` field (clip number: 0, 1, 2)
- [ ] Add `UploadedFile.duration` field (video length in seconds)
- [ ] Add `User.stripeCustomerId` field (currently commented)
- [ ] Add timestamps to Account and Session models

---

For more information:
- [Prisma Documentation](https://www.prisma.io/docs)
- [NextAuth.js Prisma Adapter](https://next-auth.js.org/adapters/prisma)
- [Architecture Overview](ARCHITECTURE.md)
- [API Documentation](API.md)
