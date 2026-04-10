"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { Languages, Brain, UserCheck, Subtitles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "~/lib/utils";

const features = [
  {
    icon: Languages,
    title: "Automatic Transcription",
    description:
      "Powered by WhisperX for ultra-accurate, multi-language transcription with word-level timestamps. Supports 99+ languages out of the box.",
  },
  {
    icon: Brain,
    title: "Smart Moment Detection",
    description:
      "Gemini AI analyzes your podcast to identify the most engaging, viral-worthy moments — the funny bits, hot takes, and drop-the-mic statements.",
  },
  {
    icon: UserCheck,
    title: "Active Speaker Detection",
    description:
      "LR-ASD technology detects who's speaking in real-time to intelligently crop and frame the active speaker for vertical video.",
  },
  {
    icon: Subtitles,
    title: "Animated Karaoke Subtitles",
    description:
      "Eye-catching, word-highlighted captions that sync perfectly with speech. Proven to increase watch time and accessibility.",
  },
];

export default function Features() {
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
    <section id="features" className="py-20 lg:py-28">
      <div ref={ref} className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-14 text-center">
          <h2 className="text-foreground text-3xl font-bold tracking-tight sm:text-4xl">
            Powered by Cutting-Edge AI
          </h2>
          <p className="text-muted-foreground mx-auto mt-4 max-w-2xl text-lg">
            Four state-of-the-art AI systems work together to transform your
            long-form content into perfectly crafted vertical clips.
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, i) => {
            const Icon = feature.icon;
            return (
              <Card
                key={feature.title}
                className={cn(
                  "group relative overflow-hidden transition-all duration-500 hover:shadow-lg hover:-translate-y-1",
                  visible
                    ? "translate-y-0 opacity-100"
                    : "translate-y-8 opacity-0"
                )}
                style={{ transitionDelay: `${i * 100}ms` }}
              >
                <CardHeader>
                  <div className="bg-primary/10 mb-3 flex h-12 w-12 items-center justify-center rounded-xl transition-transform duration-300 group-hover:scale-110">
                    <Icon className="text-primary size-6" />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-sm leading-relaxed">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </section>
  );
}
