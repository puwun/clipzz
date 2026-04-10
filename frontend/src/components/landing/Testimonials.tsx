import {
  Card,
  CardContent,
} from "~/components/ui/card";
import { Quote } from "lucide-react";

const testimonials = [
  {
    name: "Priya Sharma",
    role: "Host, The Mindful Hour Podcast",
    quote:
      "Clipzz has been a game-changer for our social media strategy. We went from spending 4 hours editing clips manually to getting perfectly cropped, subtitled clips in minutes. Our Reels engagement is up 340%.",
    avatar: "PS",
  },
  {
    name: "Marcus Chen",
    role: "Producer, TechTalk Weekly",
    quote:
      "The AI moment detection is scarily accurate. It picks up on the exact moments that make great clips — the hot takes, the laughs, the quotable lines. It's like having a video editor who truly understands content.",
    avatar: "MC",
  },
  {
    name: "Aisha Patel",
    role: "Founder, Creator Studio",
    quote:
      "We manage 12 podcast clients and Clipzz saves us over 20 hours per week. The karaoke subtitles alone are worth the price — they look better than anything we were making manually.",
    avatar: "AP",
  },
];

export default function Testimonials() {
  return (
    <section className="py-20 lg:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-14 text-center">
          <h2 className="text-foreground text-3xl font-bold tracking-tight sm:text-4xl">
            Loved by Podcasters
          </h2>
          <p className="text-muted-foreground mx-auto mt-4 max-w-2xl text-lg">
            Don&apos;t just take our word for it — hear from creators who use
            Clipzz every day.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          {testimonials.map((t) => (
            <Card
              key={t.name}
              className="group relative overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1"
            >
              <CardContent className="flex flex-col gap-4 pt-6">
                <Quote className="text-primary/20 size-8" />
                <p className="text-foreground text-sm leading-relaxed italic">
                  &ldquo;{t.quote}&rdquo;
                </p>
                <div className="mt-auto flex items-center gap-3 border-t pt-4">
                  <div className="bg-primary text-primary-foreground flex h-10 w-10 items-center justify-center rounded-full text-sm font-bold">
                    {t.avatar}
                  </div>
                  <div>
                    <p className="text-foreground text-sm font-semibold">
                      {t.name}
                    </p>
                    <p className="text-muted-foreground text-xs">{t.role}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
