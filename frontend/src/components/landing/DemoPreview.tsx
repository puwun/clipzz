"use client";

import { Play, ArrowRight, Monitor, Smartphone } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "~/lib/utils";

export default function DemoPreview() {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) setVisible(true);
      },
      { threshold: 0.2 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <section className="py-20 lg:py-28">
      <div ref={ref} className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-14 text-center">
          <h2 className="text-foreground text-3xl font-bold tracking-tight sm:text-4xl">
            See the Transformation
          </h2>
          <p className="text-muted-foreground mx-auto mt-4 max-w-2xl text-lg">
            From a full-length horizontal podcast to perfectly cropped, 
            subtitle-enhanced vertical clips — in minutes.
          </p>
        </div>

        <div
          className={cn(
            "flex flex-col items-center gap-8 transition-all duration-700 md:flex-row md:items-center md:justify-center md:gap-12",
            visible ? "translate-y-0 opacity-100" : "translate-y-8 opacity-0"
          )}
        >
          {/* Before — Horizontal */}
          <div className="flex flex-col items-center gap-3">
            <div className="text-muted-foreground mb-2 flex items-center gap-2 text-sm font-medium uppercase tracking-wider">
              <Monitor className="size-4" />
              Before
            </div>
            <div className="bg-card overflow-hidden rounded-2xl border shadow-xl">
              <div className="bg-muted relative flex aspect-video w-72 items-center justify-center sm:w-96">
                {/* Fake podcast UI */}
                <div className="flex w-full items-center gap-4 px-6">
                  {/* Speaker 1 */}
                  <div className="flex flex-col items-center gap-2">
                    <div className="bg-primary/20 flex h-14 w-14 items-center justify-center rounded-full">
                      <span className="text-primary text-lg font-bold">A</span>
                    </div>
                    <div className="bg-muted-foreground/20 h-1 w-12 rounded" />
                  </div>
                  {/* Speaker 2 */}
                  <div className="flex flex-col items-center gap-2">
                    <div className="bg-secondary flex h-14 w-14 items-center justify-center rounded-full border">
                      <span className="text-secondary-foreground text-lg font-bold">B</span>
                    </div>
                    <div className="bg-muted-foreground/20 h-1 w-12 rounded" />
                  </div>
                  {/* Waveform */}
                  <div className="flex flex-1 items-end gap-0.5">
                    {Array.from({ length: 24 }).map((_, i) => (
                      <div
                        key={i}
                        className="bg-primary/30 w-1.5 rounded-sm"
                        style={{
                          height: `${8 + Math.sin(i * 0.7) * 14 + Math.random() * 10}px`,
                        }}
                      />
                    ))}
                  </div>
                </div>

                {/* Play overlay */}
                <div className="bg-primary/10 absolute inset-0 flex items-center justify-center opacity-0 transition-opacity hover:opacity-100">
                  <div className="bg-primary/90 flex h-12 w-12 items-center justify-center rounded-full">
                    <Play className="text-primary-foreground size-5" />
                  </div>
                </div>
              </div>
              <div className="px-4 py-3">
                <p className="text-foreground text-sm font-medium">Full Episode • 1:24:35</p>
                <p className="text-muted-foreground text-xs">16:9 Landscape</p>
              </div>
            </div>
          </div>

          {/* Arrow */}
          <div className="flex flex-col items-center gap-2">
            <div className="bg-primary/10 flex h-14 w-14 items-center justify-center rounded-full">
              <ArrowRight className="text-primary size-6 max-md:rotate-90" />
            </div>
            <span className="text-primary text-xs font-semibold uppercase tracking-wider">
              AI Magic
            </span>
          </div>

          {/* After — Vertical Clips */}
          <div className="flex flex-col items-center gap-3">
            <div className="text-muted-foreground mb-2 flex items-center gap-2 text-sm font-medium uppercase tracking-wider">
              <Smartphone className="size-4" />
              After
            </div>
            <div className="flex gap-4">
              {[
                { speaker: "A", time: "0:42", label: "Hot Take", subtitle: ["That", "changes", "everything"] },
                { speaker: "B", time: "0:58", label: "Funny Moment", subtitle: ["Wait", "hold", "on", "what"] },
                { speaker: "C", time: "0:35", label: "Key Insight", subtitle: ["Here's", "the", "real", "truth"] },
              ].map((clip, i) => (
                <div
                  key={i}
                  className={cn(
                    "bg-card overflow-hidden rounded-xl border shadow-lg transition-all duration-500",
                    visible
                      ? "translate-y-0 opacity-100"
                      : "translate-y-4 opacity-0"
                  )}
                  style={{ transitionDelay: `${400 + i * 150}ms` }}
                >
                  <div className="bg-muted relative flex h-44 w-24 flex-col items-center justify-center gap-2 sm:h-52 sm:w-28">
                    {/* Speaker */}
                    <div className="bg-primary/20 flex h-10 w-10 items-center justify-center rounded-full">
                      <span className="text-primary text-sm font-bold">
                        {clip.speaker}
                      </span>
                    </div>

                    {/* Subtitle bar mockup */}
                    <div className="absolute bottom-3 left-2 right-2">
                      <div className="rounded-md bg-black/60 px-2 py-1">
                        <div className="flex gap-0.5 flex-wrap justify-center">
                          {clip.subtitle.map((word, wi) => (
                            <span
                              key={wi}
                              className={cn(
                                "text-[7px] font-bold sm:text-[8px]",
                                wi <= 1 ? "text-yellow-400" : "text-white"
                              )}
                            >
                              {word}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="px-2 py-2">
                    <p className="text-foreground text-[10px] font-medium sm:text-xs">
                      {clip.label}
                    </p>
                    <p className="text-muted-foreground text-[9px] sm:text-[10px]">
                      {clip.time} • 9:16
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
