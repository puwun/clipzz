"use server";
import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { revalidatePath } from "next/cache";
import { env } from "~/env";
import { inngest } from "~/inngest/client";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

   


export async function processVideo(uploadedFileId: string) {

    const uploadedVideo = await db.uploadedFile.findUniqueOrThrow({
        where: { id: uploadedFileId },
        select: {
           uploaded: true,
            id: true,
            userId: true,
        },
    });

    // If the video is already processed, skip
    // This is to prevent re-processing if the function is called multiple times
    if(uploadedVideo.uploaded) return;


    // Send the event to Inngest for processing
    // This will trigger the serverless function to process the video
    await inngest.send({
        name: "process-video-events",
        data: {
            uploadedFileId: uploadedVideo.id,
            userId: uploadedVideo.userId,
        },
    });


    // Update the uploadedFile record to mark it as processed
    await db.uploadedFile.update({
        where: { id: uploadedVideo.id },
        data: {
            uploaded: true,
        },
    });


    // Revalidate the dashboard path to reflect the changes
    // This will ensure that the dashboard shows the updated state from the database
    revalidatePath("/dashboard");   

} 




export async function getClipPlayUrl(
  clipId: string,
): Promise<{ succes: boolean; url?: string; error?: string }> {
  const session = await auth();
  if (!session?.user?.id) {
    return { succes: false, error: "Unauthorized" };
  }

  try {
    const clip = await db.clip.findUniqueOrThrow({
      where: {
        id: clipId,
        userId: session.user.id,
      },
    });

    const s3Client = new S3Client({
      region: env.AWS_REGION,
      credentials: {
        accessKeyId: env.AWS_ACCESS_KEY_ID,
        secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
      },
    });

    const command = new GetObjectCommand({
      Bucket: env.S3_BUCKET_NAME,
      Key: clip.s3Key,
    });

    const signedUrl = await getSignedUrl(s3Client, command, {
      expiresIn: 3600,
    });

    return { succes: true, url: signedUrl };
  } catch (error) {
      console.log("error generating URL", error)
    return { succes: false, error: "Failed to generate play URL." };
  }
}