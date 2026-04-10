"use client";

import Link from "next/link";
import { Button } from "~/components/ui/button";
import { Play, Sparkles, ArrowRight } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

const CLIPS = [
  { id: 1, label: "Hot Take", time: "0:42" },
  { id: 2, label: "Funny Bit", time: "0:58" },
  { id: 3, label: "Key Insight", time: "0:35" },
  { id: 4, label: "Viral Hook", time: "0:51" },
  { id: 5, label: "Best Quote", time: "0:39" },
  { id: 6, label: "Mic Drop", time: "0:47" },
];

export default function Hero() {
  const [visible, setVisible] = useState(false);
  // Index of the first visible clip (we show 2 on desktop, 3 on mobile)
  const [startIdx, setStartIdx] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 100);
    return () => clearTimeout(t);
  }, []);

  const advance = useCallback(() => {
    if (isTransitioning) return;
    setIsTransitioning(true);
    // After the CSS transition completes (400ms), update index and reset
    setTimeout(() => {
      setStartIdx((prev) => (prev + 1) % CLIPS.length);
      setIsTransitioning(false);
    }, 400);
  }, [isTransitioning]);

  // Auto-advance every 3 seconds
  useEffect(() => {
    timerRef.current = setInterval(advance, 3000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [advance]);

  const handleClick = () => {
    // Reset the auto-advance timer on manual click
    if (timerRef.current) clearInterval(timerRef.current);
    advance();
    timerRef.current = setInterval(advance, 3000);
  };

  const getClip = (offset: number) => CLIPS[(startIdx + offset) % CLIPS.length]!;

  return (
    <section className="relative flex min-h-[90vh] items-center justify-center overflow-hidden pt-16">
      {/* Background decorative elements */}
      <div className="pointer-events-none absolute inset-0">
        <div className="bg-primary/5 absolute top-1/4 -left-32 h-96 w-96 rounded-full blur-3xl" />
        <div className="bg-primary/8 absolute -right-32 bottom-1/4 h-96 w-96 rounded-full blur-3xl" />
        <div className="bg-accent/10 absolute top-1/2 left-1/2 h-64 w-64 -translate-x-1/2 -translate-y-1/2 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          {/* Text Content */}
          <div
            className={`flex flex-col gap-6 transition-all duration-700 ${visible ? "translate-y-0 opacity-100" : "translate-y-8 opacity-0"}`}
          >
            <div className="bg-primary/10 text-primary inline-flex w-fit items-center gap-2 rounded-full px-4 py-1.5 text-sm font-medium">
              <Sparkles className="size-4" />
              AI-Powered Podcast Clipping
            </div>

            <h1 className="text-foreground text-4xl leading-tight font-extrabold tracking-tight sm:text-5xl lg:text-6xl">
              Turn Your Podcasts Into{" "}
              <span className="text-primary">Viral Clips</span> with AI
            </h1>

            <p className="text-muted-foreground max-w-xl text-lg leading-relaxed">
              Clipzz transforms long-form podcast videos into scroll-stopping
              vertical clips for TikTok, Reels, and Shorts — automatically.
              Just upload, and let AI do the magic.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <Button size="lg" asChild className="gap-2 text-base">
                <Link href="/signup">
                  Get Started Free
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button variant="outline" size="lg" asChild className="gap-2 text-base">
                <a href="#how-it-works">
                  <Play className="size-4" />
                  See How It Works
                </a>
              </Button>
            </div>

            <p className="text-muted-foreground text-sm">
              ✨ 10 free credits on signup — no card required
            </p>
          </div>

          {/* Visual Mockup */}
          <div
            className={`flex justify-center transition-all delay-200 duration-700 ${visible ? "translate-y-0 opacity-100" : "translate-y-8 opacity-0"}`}
          >
            <div className="relative">
              {/* Horizontal video frame */}
              <div className="bg-card relative overflow-hidden rounded-2xl border shadow-xl">
                <div className="bg-muted flex aspect-video w-80 items-center justify-center sm:w-96">
                  <div className="flex flex-col items-center gap-3 p-6">
                    <div className="bg-primary/20 flex h-16 w-16 items-center justify-center rounded-full">
                      <Play className="text-primary size-8" />
                    </div>
                    <div className="text-center">
                      <p className="text-foreground text-sm font-semibold">Full Podcast Episode</p>
                      <p className="text-muted-foreground text-xs">1:24:35 • Horizontal</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Clickable Arrow */}
              <button
                onClick={handleClick}
                aria-label="Next clip"
                className="absolute -right-4 top-1/3 z-20 flex h-10 w-10 -translate-y-1/2 translate-x-full cursor-pointer items-center justify-center rounded-full transition-transform hover:scale-110 active:scale-95 max-lg:hidden"
              >
                <div className="animate-pulse">
                  <ArrowRight className="text-primary size-8" />
                </div>
              </button>

              {/* Desktop: Infinite clip carousel (shows 2 cards) */}
              <div className="absolute -bottom-6 -right-6 z-10 max-lg:hidden lg:-right-24">
                <div className="relative flex gap-3 overflow-hidden" style={{ width: "11rem" }}>
                  {/* We render 3 cards: the exiting one + 2 visible + the entering one */}
                  {[0, 1, 2].map((offset) => {
                    const clip = getClip(offset);
                    let transform = "";
                    let opacity = 1;

                    if (isTransitioning) {
                      if (offset === 0) {
                        // First card: slide out left & fade
                        transform = "translateX(-100%)";
                        opacity = 0;
                      } else {
                        // Others: slide left by one slot
                        transform = "translateX(calc(-100% - 0.75rem))";
                      }
                    }

                    return (
                      <div
                        key={`${startIdx}-${offset}`}
                        className="bg-card shrink-0 overflow-hidden rounded-xl border shadow-lg"
                        style={{
                          transition: isTransitioning
                            ? "transform 400ms ease-in-out, opacity 400ms ease-in-out"
                            : "none",
                          transform,
                          opacity,
                          width: "5rem",
                        }}
                      >
                        <div className="bg-muted flex h-36 w-20 flex-col items-center justify-center gap-1 p-2">
                          <div className="bg-primary/20 flex h-8 w-8 items-center justify-center rounded-full">
                            <Play className="text-primary size-3" />
                          </div>
                          <p className="text-foreground text-[8px] font-semibold">
                            {clip.label}
                          </p>
                          <p className="text-muted-foreground text-[7px]">
                            {clip.time} • 9:16
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Mobile: Infinite clip carousel (shows 3 cards) */}
              <div className="mt-6 lg:hidden">
                <div className="flex items-center justify-center gap-2">
                  <div className="relative flex gap-3 overflow-hidden" style={{ width: "14.5rem" }}>
                    {[0, 1, 2, 3].map((offset) => {
                      const clip = getClip(offset);
                      let transform = "";
                      let opacity = 1;

                      if (isTransitioning) {
                        if (offset === 0) {
                          transform = "translateX(-100%)";
                          opacity = 0;
                        } else {
                          transform = "translateX(calc(-100% - 0.75rem))";
                        }
                      }

                      return (
                        <div
                          key={`m-${startIdx}-${offset}`}
                          className="bg-card shrink-0 overflow-hidden rounded-xl border shadow-lg"
                          style={{
                            transition: isTransitioning
                              ? "transform 400ms ease-in-out, opacity 400ms ease-in-out"
                              : "none",
                            transform,
                            opacity,
                            width: "4rem",
                          }}
                        >
                          <div className="bg-muted flex h-28 w-16 flex-col items-center justify-center gap-1 p-2">
                            <div className="bg-primary/20 flex h-6 w-6 items-center justify-center rounded-full">
                              <Play className="text-primary size-3" />
                            </div>
                            <p className="text-foreground text-[7px] font-semibold">
                              {clip.label}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <button
                    onClick={handleClick}
                    aria-label="Next clip"
                    className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-full transition-transform hover:scale-110 active:scale-95"
                  >
                    <ArrowRight className="text-primary size-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
