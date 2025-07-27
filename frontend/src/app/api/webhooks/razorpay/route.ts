import { buffer } from "micro";
import crypto from "crypto";
import { db } from "~/server/db";
import { env } from "~/env";

export const config = {
  api: {
    bodyParser: false,
  },
};

export async function POST(req: Request) {
  const rawBody = await req.text();
  const signature = req.headers.get("x-razorpay-signature");

  const expectedSignature = crypto
    .createHmac("sha256", env.RAZORPAY_WEBHOOK_SECRET)
    .update(rawBody)
    .digest("hex");

  if (signature !== expectedSignature) {
    console.error("Invalid signature");
    return new Response(JSON.stringify({ error: "Invalid signature" }), { status: 400 });
  }

  const event = JSON.parse(rawBody);
  console.log("Received Razorpay webhook event:", event);
  if (event.event === "order.paid") {
    const order = event.payload.order.entity;
    const { priceId, userId } = order.notes || {};

    console.log("Processing order.paid for order:", order);

    const creditMap: Record<string, number> = {
      small: 50,
      medium: 150,
      large: 500,
    };

    const creditsToAdd = creditMap[priceId];

    if (!creditsToAdd || !userId) {
      console.error("Missing priceId or userId");
      return new Response(JSON.stringify({ error: "Missing priceId or userId" }), { status: 400 });
    }
    console.log(`Adding ${creditsToAdd} credits to user ${userId}`);
    await db.user.update({
      where: { id: userId },
      data: { credits: { increment: creditsToAdd } },
    });

    console.log(`Added ${creditsToAdd} credits to user ${userId}`);
  }

  return new Response(JSON.stringify({ success: true }), { status: 200 });
}