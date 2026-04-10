import { type Metadata } from "next";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { Brain, Zap, Globe, Heart } from "lucide-react";

export const metadata: Metadata = {
  title: "About",
  description:
    "Learn about Clipzz — the AI-powered podcast clip generator built to help content creators grow their audience with viral short-form content.",
};

const values = [
  {
    icon: Brain,
    title: "AI-First Approach",
    description:
      "We leverage cutting-edge AI models — WhisperX, Gemini, and LR-ASD — to deliver results that rival professional video editors.",
  },
  {
    icon: Zap,
    title: "Speed & Simplicity",
    description:
      "One upload, multiple clips. No learning curve, no complex settings. We believe powerful tools should be effortless to use.",
  },
  {
    icon: Globe,
    title: "Global Accessibility",
    description:
      "With support for 99+ languages and affordable credit-based pricing, Clipzz is built for creators everywhere, not just the biggest studios.",
  },
  {
    icon: Heart,
    title: "Creator-Centric",
    description:
      "Every feature we build starts with one question — will this help podcasters grow? If it doesn't directly serve creators, we don't build it.",
  },
];

const team = [
  { name: "Arjun Mehta", role: "Founder & CEO", initials: "AM" },
  { name: "Sneha Kapoor", role: "Lead AI Engineer", initials: "SK" },
  { name: "Daniel Park", role: "Full-Stack Developer", initials: "DP" },
  { name: "Rina Takahashi", role: "Product Designer", initials: "RT" },
];

export default function AboutPage() {
  return (
    <div className="pt-24">
      {/* Hero */}
      <section className="py-16 text-center">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <h1 className="text-foreground text-4xl font-bold tracking-tight sm:text-5xl">
            About Clipzz
          </h1>
          <p className="text-muted-foreground mx-auto mt-6 max-w-2xl text-lg leading-relaxed">
            Clipzz was born from a simple frustration — creating short-form
            clips from long podcasts is tedious, time-consuming, and expensive.
            We built Clipzz to solve this with AI, making it accessible to every
            podcaster, regardless of their technical skill or budget.
          </p>
        </div>
      </section>

      {/* Mission */}
      <section className="bg-muted/30 py-16">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-foreground text-2xl font-bold sm:text-3xl">
              Our Mission
            </h2>
            <p className="text-muted-foreground mt-4 text-lg leading-relaxed">
              To democratize short-form content creation for podcasters
              worldwide. We believe every podcast has viral moments waiting to
              be discovered — our AI just makes it effortless to find and share them.
            </p>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="py-16">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <h2 className="text-foreground mb-10 text-center text-2xl font-bold sm:text-3xl">
            What We Stand For
          </h2>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {values.map((v) => {
              const Icon = v.icon;
              return (
                <Card key={v.title} className="text-center">
                  <CardHeader>
                    <div className="bg-primary/10 mx-auto flex h-12 w-12 items-center justify-center rounded-xl">
                      <Icon className="text-primary size-6" />
                    </div>
                    <CardTitle className="text-lg">{v.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground text-sm leading-relaxed">
                      {v.description}
                    </p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* Technology */}
      <section className="bg-muted/30 py-16">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <h2 className="text-foreground mb-6 text-center text-2xl font-bold sm:text-3xl">
            Our Technology
          </h2>
          <div className="bg-card rounded-xl border p-6 sm:p-8">
            <div className="space-y-4">
              <div>
                <h3 className="text-foreground font-semibold">WhisperX Transcription</h3>
                <p className="text-muted-foreground mt-1 text-sm">
                  State-of-the-art speech-to-text with word-level timestamps and
                  speaker diarization across 99+ languages.
                </p>
              </div>
              <div>
                <h3 className="text-foreground font-semibold">Gemini AI Moment Detection</h3>
                <p className="text-muted-foreground mt-1 text-sm">
                  Google&apos;s Gemini analyzes transcripts to identify
                  high-engagement moments — hot takes, humor, insights, and
                  quotable lines.
                </p>
              </div>
              <div>
                <h3 className="text-foreground font-semibold">LR-ASD Speaker Detection</h3>
                <p className="text-muted-foreground mt-1 text-sm">
                  Light-weight Region-based Active Speaker Detection identifies
                  and frames the active speaker for optimal vertical video
                  cropping.
                </p>
              </div>
              <div>
                <h3 className="text-foreground font-semibold">Karaoke Subtitle Engine</h3>
                <p className="text-muted-foreground mt-1 text-sm">
                  Word-by-word highlighted captions that sync perfectly with
                  speech, dramatically increasing watch time and accessibility.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="py-16">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <h2 className="text-foreground mb-10 text-center text-2xl font-bold sm:text-3xl">
            Meet the Team
          </h2>
          <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
            {team.map((t) => (
              <div key={t.name} className="flex flex-col items-center gap-3 text-center">
                <div className="bg-primary text-primary-foreground flex h-16 w-16 items-center justify-center rounded-full text-lg font-bold">
                  {t.initials}
                </div>
                <div>
                  <p className="text-foreground text-sm font-semibold">{t.name}</p>
                  <p className="text-muted-foreground text-xs">{t.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
