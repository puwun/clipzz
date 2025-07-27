"use server";

import { redirect } from "next/navigation";
import Stripe from "stripe";
import { env } from "~/env";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

const stripe = new Stripe(env.STRIPE_SECRET_KEY, {
  apiVersion: "2025-06-30.basil",
});

export type PriceId = "small" | "medium" | "large";

const PRICE_IDS: Record<PriceId, string> = {
  small: env.STRIPE_SMALL_CREDIT_PACK,
  medium: env.STRIPE_MEDIUM_CREDIT_PACK,
  large: env.STRIPE_LARGE_CREDIT_PACK,
};

export async function createCheckoutSession(priceId: PriceId) {
  const serverSession = await auth();

  const user = await db.user.findUniqueOrThrow({
    where: {
      id: serverSession?.user.id,
    },
    select: { stripeCustomerId: true },
  });

  if (!user.stripeCustomerId) {
    throw new Error("User has no stripeCustomerId");
  }

  const session = await stripe.checkout.sessions.create({
    line_items: [{ price: PRICE_IDS[priceId], quantity: 1 }],
    // customer: user.stripeCustomerId,
    mode: "payment",
    success_url: `${env.BASE_URL}/dashboard?success=true`,
  });

  if (!session.url) {
    throw new Error("Failed to create session URL");
  }

  redirect(session.url);
}


/*
working stripe URL:

url -X POST http://localhost:3000/api/webhooks/stripe   -H "Content-Type: application/json"   -d '{"type": "checkout.session.completed", "data": {"object": {"id": "cs_test_a1ybuPhzeY5JUMVvzqMFUAXASO88e3xhAfZEBJhsRMRipJZXicnigtWqZ4", "customer": "cus_Ske9m3Q0lemYQx"}}}'

  but only through CLI


--option to select number of clips to Generate
--after clicking on one of the entries in queue status, it should redirect to my clips for those particular generated clips
--sort and segregate my clips based on some paramter (date ig)

  */



