  import { env } from "~/env";
import { inngest } from "./client";
import { db } from "~/server/db";
import { ListObjectsV2Command, S3Client } from "@aws-sdk/client-s3";

export const processVideo = inngest.createFunction(
  {
    id: "process-video",
    retries: 1,
    concurrency: {        // processing 1 req for each user at a time
      limit: 1,
      key: "event.data.userId"
    }
  },
  { event: "process-video-events" },
  async ({ event, step }) => {

    const { uploadedFileId, numClips } = event.data as {uploadedFileId:  string, userId: string, numClips: number};

    const { userId, s3Key, credits } = await step.run("check-credits", async () => {
      const uploadedFile = await db.uploadedFile.findUniqueOrThrow(
        {
          where: {
            id: uploadedFileId
          },
          select: {
            user: {
              select: {
                id: true,
                credits: true,
              },
            },
            s3Key: true,
          },
        }
      )

      return { userId: uploadedFile.user.id, s3Key: uploadedFile.s3Key, credits: uploadedFile.user.credits };
    })


    if (credits > 0) {

      await step.run("set-status-processing", async () => {
        await db.uploadedFile.update({
          where: { id: uploadedFileId },
          data: { status: "processing" },
        });
      });


      await step.run("call-modal-endpoint", async () => {
        await fetch(env.PROCESS_VIDEO_ENDPOINT, {
          method: "POST",
          body: JSON.stringify({
            s3_key: s3Key,
            num_clips: numClips
          }),
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}`,
          },
        })
      });


      const { clipsFound } = await step.run("create-clips-in-db", async () => {

        const folderPrefix = s3Key.split("/")[0]!;
        const s3Objects = await getS3ObjectsByPrefix(folderPrefix);

        const clipKeys = s3Objects.filter(
          (key): key is string =>
            key !== undefined && !key.endsWith("original.mp4"),
        );

        if (clipKeys.length > 0) {

          await db.clip.createMany({
            data: clipKeys.map((key) => ({
              userId,
              s3Key: key,
              uploadedFileId,
            })),
          });
        }

        return { clipsFound: clipKeys.length };
      });

      await step.run("decrement-credits", async () => {
        await db.user.update({
          where: { id: userId },
          data: { credits: { decrement: Math.min(credits, clipsFound) } },    // if user has 2 creds and uses compute of 5 creds, 2 will be deducted but processing of 5 will be done (user gets a bit more than he paid for ), another option is to check if sufficient creds before processing 
        });
      });

      await step.run("set-status-processed", async () => {
        await db.uploadedFile.update({
          where: { id: uploadedFileId },
          data: { status: "processed" },
        });
      });


    } else {
      await step.run("set-status-no-credits", async () => {
        await db.uploadedFile.update({
          where: { id: uploadedFileId },
          data: { status: "no-credits" },
        });
      });
    }
  },
);




async function getS3ObjectsByPrefix(prefix: string) {
  const s3Client = new S3Client({
    region: env.AWS_REGION,
    credentials: {
      accessKeyId: env.AWS_ACCESS_KEY_ID,
      secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
    },
  });


  const listCmd = new ListObjectsV2Command({
    Bucket: env.S3_BUCKET_NAME,
    Prefix: prefix,
  });

  const response = await s3Client.send(listCmd);

  // Ensure response.Contents is defined before mapping
  if (!response.Contents) {
    return [];
  }
  return response.Contents?.map((obj) => obj.Key).filter(Boolean ?? []);
}