import { type Metadata } from "next";
import PublicNav from "~/components/landing/PublicNav";
import Footer from "~/components/landing/Footer";

export const metadata: Metadata = {
  title: {
    default: "Clipzz — AI-Powered Podcast Clip Generator",
    template: "%s | Clipzz",
  },
  description:
    "Transform long-form podcast videos into viral-ready vertical clips for TikTok, Reels, and Shorts with AI-powered transcription, moment detection, and karaoke subtitles.",
  openGraph: {
    type: "website",
    siteName: "Clipzz",
    title: "Clipzz — AI-Powered Podcast Clip Generator",
    description:
      "Transform long-form podcast videos into viral-ready vertical clips for TikTok, Reels, and Shorts.",
  },
  twitter: {
    card: "summary_large_image",
    title: "Clipzz — AI-Powered Podcast Clip Generator",
    description:
      "Transform long-form podcast videos into viral-ready vertical clips for TikTok, Reels, and Shorts.",
  },
};

export default function MarketingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="bg-background text-foreground flex min-h-screen flex-col">
      <PublicNav />
      <main className="flex-1">{children}</main>
      <Footer />
    </div>
  );
}
