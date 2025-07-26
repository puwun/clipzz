// stripe listen --forward-to localhost:3000/api/webhooks/stripe

import { NextResponse } from "next/server";
import Stripe from "stripe";
import { env } from "~/env";
import { db } from "~/server/db";
// import getRawBody from "raw-body";

console.log("Initializing Stripe webhook handler...");

const stripe = new Stripe(env.STRIPE_SECRET_KEY, {
  apiVersion: "2025-06-30.basil",
});

console.log("Stripe initialized with API version:", stripe);

const webhookSecret = env.STRIPE_WEBHOOK_SECRET;

console.log("Stripe webhook secret:", webhookSecret);

export async function POST(req: Request) {
  console.log("==================================");
  console.log("Received webhook event at:", new Date().toISOString());
  console.log("Request headers:", Object.fromEntries(req.headers.entries()));
  try {
    const body = await req.text();
    console.log("req.headers:", req.headers);
    const signature = req.headers.get("stripe-signature") || "";

  console.log("Webhook body:", body);
    console.log("Webhook signature:", signature);
    console.log("Webhook secret:", webhookSecret);

    let event: Stripe.Event;

    console.log("Verifying webhook signature...", body);

    try {
      event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
    } catch (error) {
      console.error("Webhook signature verification failed", error);
      return new NextResponse("Webhook signature verification failed", {
        status: 400,
      });
    }
    // try {
    //   event = JSON.parse(body) as Stripe.Event;
    //   console.log("Parsed event:", event);
    // } catch (error) {
    //   console.error("Failed to parse event:", error);
    //   return new NextResponse("Invalid payload", { status: 400 });
    // }


    if (event.type === "checkout.session.completed") {
      const session = event.data.object;
      const customerId = session.customer as string;


      console.log("session is completed", session.id, "for customer", customerId);

      const retreivedSession = await stripe.checkout.sessions.retrieve(
        session.id,
        { expand: ["line_items"] },
      );

      console.log("Retrieved session:", retreivedSession);

      const lineItems = retreivedSession.line_items;
      console.log("Line items:", lineItems);

      if (lineItems && lineItems.data.length > 0) {
        const priceId = lineItems.data[0]?.price?.id ?? undefined;

        if (priceId) {
          let creditsToAdd = 0;

          if (priceId === env.STRIPE_SMALL_CREDIT_PACK) {
            creditsToAdd = 50;
          } else if (priceId === env.STRIPE_MEDIUM_CREDIT_PACK) {
            creditsToAdd = 150;
          } else if (priceId === env.STRIPE_LARGE_CREDIT_PACK) {
            creditsToAdd = 500;
          }

          console.log(`Adding ${creditsToAdd} credits for customer ${customerId} for price ${priceId}`);
          
          
          await db.user.update({
            where: { stripeCustomerId: customerId },
            data: {
              credits: {
                increment: creditsToAdd,
              },
            },
          });
        }
      }
    }

    return new NextResponse(null, { status: 200 });
  } catch (error) {
    console.error("Error processing webhook:", error);
    return new NextResponse("Webhook error", { status: 500 });
  }
}