"use client";

import type { VariantProps } from "class-variance-authority";
import { ArrowLeftIcon, CheckIcon } from "lucide-react";
import Link from "next/link";
import { createCheckoutSession, type PriceId } from "~/actions/razorpay";
import { Button } from "~/components/ui/button";
import type { buttonVariants } from "~/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { cn } from "~/lib/utils";
import Script from "next/script";
import { useState } from "react";
import { getCurrentUser } from "~/actions/razorpay";

interface PricingPlan {
  title: string;
  price: string;
  description: string;
  features: string[];
  buttonText: string;
  buttonVariant: VariantProps<typeof buttonVariants>["variant"];
  isPopular?: boolean;
  savePercentage?: string;
  priceId: PriceId;
}

const plans: PricingPlan[] = [
  {
    title: "Small Pack",
    price: "₹830",
    description: "Perfect for occasional podcast creators",
    features: ["50 credits", "No expiration", "Download all clips"],
    buttonText: "Buy 50 credits",
    buttonVariant: "outline",
    priceId: "small",
  },
  {
    title: "Medium Pack",
    price: "₹2075",
    description: "Best value for regular podcasters",
    features: ["150 credits", "No expiration", "Download all clips"],
    buttonText: "Buy 150 credits",
    buttonVariant: "default",
    isPopular: true,
    savePercentage: "Save 17%",
    priceId: "medium",
  },
  {
    title: "Large Pack",
    price: "₹5810",
    description: "Ideal for podcast studios and agencies",
    features: ["500 credits", "No expiration", "Download all clips"],
    buttonText: "Buy 500 credits",
    buttonVariant: "outline",
    isPopular: false,
    savePercentage: "Save 30%",
    priceId: "large",
  },
];

function PricingCard({ plan }: { plan: PricingPlan }) {
  const [isLoading, setIsLoading] = useState(false);

  const handleCheckout = async () => {
    console.log("reached 01");
    setIsLoading(true);
    console.log("reached 02");
    try {
      console.log("reached 03");
      console.log("Initiating Razorpay checkout for plan:", plan.priceId);
      const user = await getCurrentUser(); // Fetch user ID
      console.log("reached 04");
      if (!user) {
        console.error("No user logged in");
        alert("Please log in to make a purchase.");
        setIsLoading(false);
        return;
      }
      
      console.log("reached 05");
      console.log("before creating checkout session");
      const order = await createCheckoutSession(plan.priceId);
      console.log("Order details received:", order);

      if (!(window as any).Razorpay) {
        console.error("Razorpay script not loaded");
        alert("Payment service unavailable. Please try again later.");
        setIsLoading(false);
        return;
      }

      const options = {
        key: order.keyId,
        order_id: order.orderId,
        amount: order.amount,
        currency: order.currency,
        name: order.name,
        description: order.description,
        prefill: { email: order.email },
        theme: { color: "#3399cc" },
        notes: { priceId: plan.priceId, userId: user.id },
        mode: "test",
        handler: function (response: any) {
          console.log("Payment successful:", response);
          window.location.href = "/dashboard?success=true";
        },
        modal: {
          ondismiss: function () {
            console.log("Razorpay checkout dismissed");
            setIsLoading(false);
          },
        },
      };

      console.log("Razorpay options:", options);

      const rzp = new window.Razorpay(options);
      rzp.on("payment.failed", function (response: any) {
        console.error("Payment failed:", response.error);
        alert(`Payment failed: ${response.error.description}`);
        setIsLoading(false);
      });
      rzp.open();
    } catch (error) {
      console.error("Checkout error:", error);
      alert("Failed to initiate payment. Please try again.");
      setIsLoading(false);
    }
  };

  return (
    <Card
      className={cn(
        "relative flex flex-col",
        plan.isPopular && "border-primary border-2"
      )}
    >
      {plan.isPopular && (
        <div className="bg-primary text-primary-foreground absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 transform rounded-full px-3 py-1 text-sm font-medium whitespace-nowrap">
          Most Popular
        </div>
      )}
      <CardHeader className="flex-1">
        <CardTitle>{plan.title}</CardTitle>
        <div className="text-4xl font-bold">{plan.price}</div>
        {plan.savePercentage && (
          <p className="text-sm font-medium text-green-600">
            {plan.savePercentage}
          </p>
        )}
        <CardDescription>{plan.description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <ul className="text-muted-foreground space-y-2 text-sm">
          {plan.features.map((feature, index) => (
            <li key={index} className="flex items-center gap-2">
              <CheckIcon className="text-primary size-4" />
              {feature}
            </li>
          ))}
        </ul>
      </CardContent>
      <CardFooter>
        <Button
          variant={plan.buttonVariant}
          className="w-full"
          onClick={() => {
            console.log("Button clicked for plan:", plan.title);
            handleCheckout();
          }}  
          disabled={isLoading}
          type="button"
        >
          {isLoading ? "Processing..." : plan.buttonText}
        </Button>
      </CardFooter>
    </Card>
  );
}

export default function BillingPage() {
  return (
    <>
      <Script
        src="https://checkout.razorpay.com/v1/checkout.js"
        strategy="lazyOnload"
        onLoad={() => console.log("Razorpay script loaded successfully")}
        onError={(e) => console.error("Razorpay script failed to load:", e)}
      />
      <div className="mx-auto flex flex-col space-y-8 px-4 py-12">
        <div className="relative flex items-center justify-center gap-4">
          <Button
            className="absolute top-0 left-0"
            variant="outline"
            size="icon"
            asChild
          >
            <Link href="/dashboard">
              <ArrowLeftIcon className="size-4" />
            </Link>
          </Button>
          <div className="space-y-2 text-center">
            <h1 className="text-2xl font-bold tracking-tight sm:text-4xl">
              Buy Credits
            </h1>
            <p className="text-muted-foreground">
              Purchase credits to generate more podcast clips. The more credits
              you buy, the better the value.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
          {plans.map((plan) => (
            <PricingCard key={plan.title} plan={plan} />
          ))}
        </div>

        <div className="bg-muted/50 rounded-lg p-6">
          <h3 className="mb-4 text-lg font-semibold">How credits work</h3>
          <ul className="text-muted-foreground list-disc space-y-2 pl-5 text-sm">
            <li>1 credit = 1 minute of podcast processing</li>
            <li>The program will create around 1 clip per 5 minutes of podcast</li>
            <li>Credits never expire and can be used anytime</li>
            <li>Longer podcasts require more credits based on duration</li>
            <li>All packages are one-time purchases (not subscription)</li>
          </ul>
        </div>
      </div>
    </>
  );
}