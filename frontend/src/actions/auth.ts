"use server";

import { hashPassword } from "~/lib/auth";
import { signupSchema, type SignupFormValues } from "~/schemas/auth";
import { db } from "~/server/db";
import Stripe from "stripe";
import { env } from "~/env";
import axios from "axios";

type SignupResult = {
  success: boolean;
  error?: string;
};




export async function signUp(data: SignupFormValues): Promise<SignupResult> {
  const validationResult = signupSchema.safeParse(data);
  if (!validationResult.success) {
    return {
      success: false,
      error: validationResult.error.issues[0]?.message ?? "Invalid input",
    };
  }

  const { email, password } = validationResult.data;

  try {
    const existingUser = await db.user.findUnique({ where: { email } });

    if (existingUser) {
      return {
        success: false,
        error: "Email already in use",
      };
    }


    // Create Razorpay contact
    const razorpayResponse = await axios.post(
      "https://api.razorpay.com/v1/contacts",
      {
        name: email.split("@")[0], // Use email prefix as name (or customize as needed)
        email: email.toLowerCase(),
        type: "customer",
      },
      {
        auth: {
        username: env.RAZORPAY_KEY_ID,
        password: env.RAZORPAY_KEY_SECRET,
        },
      }
    );
    // console.log("----------------------------")
    // console.log("Razorpay response:", razorpayResponse);
    // console.log("----------------------------")


    // if (!razorpayResponse.ok) {
    //   const errorData = await razorpayResponse.json();
    //   console.error("Razorpay customer creation failed:", errorData);
    //   return {
    //     success: false,
    //     error: "Failed to create Razorpay customer",
    //   };
    // }
    // const stripe = new Stripe(env.STRIPE_SECRET_KEY);

    // const stripeCustomer = await stripe.customers.create({
    //   email: email.toLowerCase(),
    // });



    // Only hash and set password if provided
    console.log("reached here before stripe custID")
    
    const userData = {
      email,
      razorpayContactId: razorpayResponse.data.id,
    };

    if (password) {
      const hashedPassword = await hashPassword(password);
      userData.password = hashedPassword;
    }

    await db.user.create({
      data: userData,
    });

    return { success: true };
  } catch (error) {
      console.error("Signup error:", error);
    return { success: false, error: "An error occurred during signup" };
  }
}