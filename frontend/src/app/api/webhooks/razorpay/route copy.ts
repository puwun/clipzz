import { db } from "~/server/db";
import { env } from "~/env";
import { buffer } from "micro";
import { NextResponse } from "next/server";
import crypto from "crypto";

export const config = {
  api: {
    bodyParser: false,
  },
};

export async function POST(request: Request) {
  console.log("Received RazorpayX webhook");

  try {
    // Get raw body for signature verification
    const rawBodyBuffer = await new Response(request.body).arrayBuffer();
    const rawBody = Buffer.from(rawBodyBuffer);
    const signature = request.headers.get("x-razorpay-signature") ?? "";
    const eventId = request.headers.get("x-razorpay-event-id") ?? "";
    const webhookSecret = env.RAZORPAY_WEBHOOK_SECRET;

    if (!webhookSecret) {
      console.error("Missing RAZORPAY_WEBHOOK_SECRET");
      return NextResponse.json({ error: "Webhook secret not configured" }, { status: 400 });
    }

    console.log("Raw body:", rawBody.toString());

    // Verify signature
    const expectedSignature = crypto
      .createHmac("sha256", webhookSecret)
      .update(rawBody.toString())
      .digest("hex");

    console.log("Received signature:", signature);
    console.log("Expected signature:", expectedSignature);

    if (signature !== expectedSignature) {
      console.error("Invalid webhook signature");
      return NextResponse.json({ error: "Invalid signature" }, { status: 400 });
    }

    // Parse body
    const event = JSON.parse(rawBody.toString());
    console.log("Webhook event:", event);

    console.log("Event is:", event.event);
    // Handle events
    switch (event.event) {
        case "order.paid":
        console.log("Processing order.paid for order:", event.payload.order.entity);
        const order = event.payload.order.entity;
        const { priceId, userId } = order.notes || {};

        if (!priceId || !userId) {
          console.error("Missing priceId or userId in order notes");
          return NextResponse.json({ error: "Invalid order notes" }, { status: 400 });
        }

        const creditMap: Record<string, number> = {
          small: 50,
          medium: 150,
          large: 500,
        };

        const creditsToAdd = creditMap[priceId];
        if (!creditsToAdd) {
          console.error("Invalid priceId:", priceId);
          return NextResponse.json({ error: "Invalid priceId" }, { status: 400 });
        }

        await db.user.update({
          where: { id: userId },
          data: { credits: { increment: creditsToAdd } },
        });

        console.log(`Added ${creditsToAdd} credits to user ${userId}`);
        break;

      case "transaction.created":
        console.log("Transaction created:", event);
        // Example: Log transaction or update a transaction record
        break;

      case "payout.processed":
    // Payment Gateway: Add credits on successful payment
        console.log("Processing payout.processed for payout:", event.payload.order.entity);
        // const order_id  = event.payload.order.entity.id;

        // console.log("Order ID:", order_id);

        // const orderResponse = await fetch(`https://api.razorpay.com/v1/orders/${order_id}`, {
        //   headers: {
        //     Authorization: `Basic ${Buffer.from(`${env.RAZORPAY_KEY_ID}:${env.RAZORPAY_KEY_SECRET}`).toString("base64")}`,
        //   },
        // });

        // if (!orderResponse.ok) {
        //   console.error("Failed to fetch order:", orderResponse.statusText);
        //   return NextResponse.json({ error: "Failed to fetch order" }, { status: 500 });
        // }

        // const order = await orderResponse.json();
        // const { priceId, userId } = order.notes || {};

        // if (!priceId || !userId) {
        //   console.error("Missing priceId or userId in order notes");
        //   return NextResponse.json({ error: "Invalid order notes" }, { status: 400 });
        // }

        // const creditMap: Record<string, number> = {
        //   small: 50,
        //   medium: 150,
        //   large: 500,
        // };

        // const creditsToAdd = creditMap[priceId];
        // if (!creditsToAdd) {
        //   console.error("Invalid priceId:", priceId);
        //   return NextResponse.json({ error: "Invalid priceId" }, { status: 400 });
        // }

        // await db.user.update({
        //   where: { id: userId },
        //   data: { credits: { increment: creditsToAdd } },
        // });

        // console.log(`Added ${creditsToAdd} credits to user ${userId}`);
        break;

      case "payout.reversed":
        console.log("Payout reversed:", event);
        // Example: Handle reversal (e.g., notify user)
        break;

      case "payout.rejected":
        console.log("Payout rejected:", event);
        // Example: Log error or retry payout
        break;

      case "payout.pending":
        console.log("Payout pending:", event);
        // Example: Update payout status
        break;

      case "fund_account.validation.completed":
        console.log("Fund account validation completed:", event);
        // Example: Mark account as validated
        break;

      case "fund_account.validation.failed":
        console.log("Fund account validation failed:", event);
        // Example: Notify user to correct account details
        break;

      default:
        console.log("Unhandled event:", event.event);
    }

    return NextResponse.json({ received: true });
  } catch (error) {
    console.error("Webhook error:", error);
    return NextResponse.json({ error: "Webhook processing failed" }, { status: 500 });
  }
}