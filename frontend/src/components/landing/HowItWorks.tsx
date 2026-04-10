"use client";

import { Upload, FileAudio, Sparkles, Scissors, Download } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "~/lib/utils";

const steps = [
  {
    icon: Upload,
    title: "Upload",
    description: "Upload your podcast video in any popular format. We support MP4, MOV, AVI, and more.",
  },
  {
    icon: FileAudio,
    title: "Transcribe",
    description: "WhisperX generates a precise, timestamped transcription of every word spoken.",
  },
  {
    icon: Sparkles,
    title: "Detect Moments",
    description: "Gemini AI identifies the most viral-worthy, engaging highlights from your podcast.",
  },
  {
    icon: Scissors,
    title: "Generate Clips",
    description: "Smart cropping, active speaker detection, and karaoke subtitles are applied automatically.",
  },
  {
    icon: Download,
    title: "Download & Share",
    description: "Get vertical, ready-to-post clips optimized for TikTok, Instagram Reels, and YouTube Shorts.",
  },
];

export default function HowItWorks() {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) setVisible(true);
      },
      { threshold: 0.1 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <section id="how-it-works" className="bg-muted/30 py-20 lg:py-28">
      <div ref={ref} className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-14 text-center">
          <h2 className="text-foreground text-3xl font-bold tracking-tight sm:text-4xl">
            How It Works
          </h2>
          <p className="text-muted-foreground mx-auto mt-4 max-w-2xl text-lg">
            From full podcast to viral clips in five simple steps.
          </p>
        </div>

        <div className="relative grid gap-8 md:grid-cols-5">
          {/* Connector line (desktop) */}
          <div className="bg-border absolute top-12 right-[10%] left-[10%] hidden h-0.5 md:block" />

          {steps.map((step, i) => {
            const Icon = step.icon;
            return (
              <div
                key={step.title}
                className={cn(
                  "relative flex flex-col items-center text-center transition-all duration-500",
                  visible
                    ? "translate-y-0 opacity-100"
                    : "translate-y-6 opacity-0"
                )}
                style={{ transitionDelay: `${i * 120}ms` }}
              >
                {/* Step number + icon */}
                <div className="bg-background relative z-10 mb-4 flex h-24 w-24 flex-col items-center justify-center rounded-2xl border shadow-sm">
                  <span className="text-primary text-xs font-bold uppercase tracking-wider">
                    Step {i + 1}
                  </span>
                  <Icon className="text-primary mt-1 size-8" />
                </div>

                <h3 className="text-foreground mb-2 text-base font-semibold">
                  {step.title}
                </h3>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {step.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
