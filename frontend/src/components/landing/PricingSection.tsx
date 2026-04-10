import Link from "next/link";
import { CheckIcon } from "lucide-react";
import { Button } from "~/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { Badge } from "~/components/ui/badge";
import { cn } from "~/lib/utils";

const plans = [
  {
    title: "Small Pack",
    price: "₹830",
    description: "Perfect for occasional podcast creators",
    features: ["50 credits", "No expiration", "Download all clips"],
    buttonText: "Get Started",
    isPopular: false,
    savePercentage: null,
  },
  {
    title: "Medium Pack",
    price: "₹2,075",
    description: "Best value for regular podcasters",
    features: ["150 credits", "No expiration", "Download all clips"],
    buttonText: "Get Started",
    isPopular: true,
    savePercentage: "Save 17%",
  },
  {
    title: "Large Pack",
    price: "₹5,810",
    description: "Ideal for podcast studios and agencies",
    features: ["500 credits", "No expiration", "Download all clips"],
    buttonText: "Get Started",
    isPopular: false,
    savePercentage: "Save 30%",
  },
];

export default function PricingSection() {
  return (
    <section id="pricing" className="bg-muted/30 py-20 lg:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-14 text-center">
          <h2 className="text-foreground text-3xl font-bold tracking-tight sm:text-4xl">
            Simple, Transparent Pricing
          </h2>
          <p className="text-muted-foreground mx-auto mt-4 max-w-2xl text-lg">
            Pay per credit, no subscriptions. Buy what you need, use whenever
            you want.
          </p>
          <Badge variant="secondary" className="mt-4 px-4 py-1.5 text-sm">
            ✨ 10 free credits on signup
          </Badge>
        </div>

        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-8 md:grid-cols-3">
          {plans.map((plan) => (
            <Card
              key={plan.title}
              className={cn(
                "relative flex flex-col transition-all duration-300 hover:shadow-lg hover:-translate-y-1",
                plan.isPopular && "border-primary border-2 scale-[1.02]"
              )}
            >
              {plan.isPopular && (
                <div className="bg-primary text-primary-foreground absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 transform rounded-full px-4 py-1 text-sm font-medium whitespace-nowrap">
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
                <ul className="text-muted-foreground space-y-2.5 text-sm">
                  {plan.features.map((feature) => (
                    <li key={feature} className="flex items-center gap-2">
                      <CheckIcon className="text-primary size-4 shrink-0" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </CardContent>
              <CardFooter>
                <Button
                  variant={plan.isPopular ? "default" : "outline"}
                  className="w-full"
                  asChild
                >
                  <Link href="/signup">{plan.buttonText}</Link>
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>

        {/* How credits work */}
        <div className="bg-muted/50 mx-auto mt-12 max-w-3xl rounded-xl border p-6">
          <h3 className="text-foreground mb-4 text-lg font-semibold">
            How credits work
          </h3>
          <ul className="text-muted-foreground list-disc space-y-2 pl-5 text-sm">
            <li>1 credit = 1 minute of podcast processing</li>
            <li>
              The program will create around 1 clip per 5 minutes of podcast
            </li>
            <li>Credits never expire and can be used anytime</li>
            <li>Longer podcasts require more credits based on duration</li>
            <li>All packages are one-time purchases (not subscriptions)</li>
          </ul>
        </div>
      </div>
    </section>
  );
}
