import { type Metadata } from "next";
import Hero from "~/components/landing/Hero";
import StatsBar from "~/components/landing/StatsBar";
import Features from "~/components/landing/Features";
import HowItWorks from "~/components/landing/HowItWorks";
import DemoPreview from "~/components/landing/DemoPreview";
import PricingSection from "~/components/landing/PricingSection";
import Testimonials from "~/components/landing/Testimonials";
import FAQ from "~/components/landing/FAQ";
import Link from "next/link";
import { Button } from "~/components/ui/button";
import { ArrowRight } from "lucide-react";

export const metadata: Metadata = {
  title: "Clipzz — Turn Your Podcasts Into Viral Clips with AI",
  description:
    "Clipzz is an AI-powered platform that transforms long-form podcast videos into viral-ready vertical clips for TikTok, Instagram Reels, and YouTube Shorts. Automatic transcription, smart moment detection, and animated subtitles.",
};

export default function LandingPage() {
  return (
    <>
      <Hero />
      <StatsBar />
      <Features />
      <HowItWorks />
      <DemoPreview />
      <PricingSection />
      <Testimonials />
      <FAQ />

      {/* Final CTA Section */}
      <section className="py-20 lg:py-28">
        <div className="mx-auto max-w-4xl px-4 text-center sm:px-6 lg:px-8">
          <div className="bg-card relative overflow-hidden rounded-3xl border p-10 shadow-lg sm:p-16">
            {/* Decorative blobs */}
            <div className="pointer-events-none absolute inset-0">
              <div className="bg-primary/5 absolute -top-16 -left-16 h-48 w-48 rounded-full blur-3xl" />
              <div className="bg-primary/8 absolute -right-16 -bottom-16 h-48 w-48 rounded-full blur-3xl" />
            </div>

            <div className="relative z-10">
              <h2 className="text-foreground text-3xl font-bold tracking-tight sm:text-4xl">
                Ready to Go Viral?
              </h2>
              <p className="text-muted-foreground mx-auto mt-4 max-w-xl text-lg">
                Join hundreds of podcasters who are already using Clipzz to
                grow their audience. Start with 10 free credits — no card
                required.
              </p>
              <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                <Button size="lg" asChild className="gap-2 text-base">
                  <Link href="/signup">
                    Get Started Free
                    <ArrowRight className="size-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild className="text-base">
                  <Link href="/login">Login to Dashboard</Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
