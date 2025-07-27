"use server";

import axios, { AxiosError } from "axios";
import { env } from "~/env";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

export type PriceId = "small" | "medium" | "large";

const ITEM_IDS: Record<PriceId, string> = {
  small: env.RAZORPAY_SMALL_CREDIT_PACK,
  medium: env.RAZORPAY_MEDIUM_CREDIT_PACK,
  large: env.RAZORPAY_LARGE_CREDIT_PACK,
};

export  async function getCurrentUser() {
  const serverSession = await auth();
  if (!serverSession?.user) {
    throw new Error("User not authenticated");
  }

  console.log("Current user:", serverSession.user);
  return serverSession.user;
}


export async function createCheckoutSession(priceId: string) {
  const user = await auth();
  console.log("Creating checkout session for user:", user);
  if (!user) {
    throw new Error("Unauthorized");
  }

  const amounts: Record<string, number> = {
    small: 83000,
    medium: 207500,
    large: 581000,
  };

  if (!amounts[priceId]) {
    throw new Error("Invalid priceId");
  }

  console.log("user ID:", user.user.id);
  const response = await fetch("https://api.razorpay.com/v1/orders", {
    method: "POST",
    headers: {
      Authorization: `Basic ${Buffer.from(`${env.RAZORPAY_KEY_ID}:${env.RAZORPAY_KEY_SECRET}`).toString("base64")}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      amount: amounts[priceId],
      currency: "INR",
      notes: { priceId, userId: user.user.id },
    }),
  });


  if (!response.ok) {
    throw new Error(`Failed to create order: ${response.statusText}`);
  }

  const order = await response.json();
  console.log("Razorpay order created:", order);
  return {
    keyId: env.RAZORPAY_KEY_ID,
    orderId: order.id,
    amount: order.amount,
    currency: order.currency,
    name: "Podcast Clips",
    description: `Purchase ${priceId} credit pack`,
    email: user.user.email,
    
  };
}