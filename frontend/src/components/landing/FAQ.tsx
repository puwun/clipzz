"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "~/lib/utils";

const faqs = [
  {
    question: "What video formats does Clipzz support?",
    answer:
      "Clipzz supports all popular video formats including MP4, MOV, AVI, MKV, and WebM. Simply upload your podcast video and our system handles the rest. We recommend MP4 for the best compatibility.",
  },
  {
    question: "How long does processing take?",
    answer:
      "Processing typically takes 2-5 minutes per clip, depending on the length of your podcast and server load. A 1-hour podcast usually generates 8-12 clips and takes about 15-20 minutes total.",
  },
  {
    question: "How does the credit system work?",
    answer:
      "1 credit equals 1 minute of podcast processing. So a 60-minute podcast uses 60 credits. The program creates approximately 1 clip per 5 minutes of content. Credits never expire and all packages are one-time purchases — no subscriptions.",
  },
  {
    question: "What languages are supported?",
    answer:
      "Our WhisperX transcription engine supports 99+ languages including English, Hindi, Spanish, French, German, Japanese, Korean, Mandarin, Arabic, Portuguese, and many more. Subtitles are generated in the detected language automatically.",
  },
  {
    question: "Is my content private and secure?",
    answer:
      "Absolutely. Your videos are stored securely on AWS S3 with encryption at rest and in transit. We never share your content with third parties. You can delete your uploaded files and generated clips at any time from your dashboard.",
  },
  {
    question: "Can I customize the generated clips?",
    answer:
      "Currently, Clipzz automatically selects the best moments and applies optimal cropping, framing, and subtitles. We're working on customization features like subtitle styles, manual moment selection, and branding options — coming soon!",
  },
];

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <section id="faq" className="bg-muted/30 py-20 lg:py-28">
      <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
        <div className="mb-14 text-center">
          <h2 className="text-foreground text-3xl font-bold tracking-tight sm:text-4xl">
            Frequently Asked Questions
          </h2>
          <p className="text-muted-foreground mx-auto mt-4 max-w-2xl text-lg">
            Everything you need to know about Clipzz.
          </p>
        </div>

        <div className="space-y-3">
          {faqs.map((faq, i) => {
            const isOpen = openIndex === i;
            return (
              <div
                key={i}
                className="bg-card overflow-hidden rounded-xl border transition-all duration-200"
              >
                <button
                  type="button"
                  className="flex w-full items-center justify-between gap-4 px-6 py-4 text-left"
                  onClick={() => setOpenIndex(isOpen ? null : i)}
                  aria-expanded={isOpen}
                  id={`faq-trigger-${i}`}
                  aria-controls={`faq-content-${i}`}
                >
                  <span className="text-foreground text-sm font-medium sm:text-base">
                    {faq.question}
                  </span>
                  <ChevronDown
                    className={cn(
                      "text-muted-foreground size-5 shrink-0 transition-transform duration-200",
                      isOpen && "rotate-180"
                    )}
                  />
                </button>
                <div
                  id={`faq-content-${i}`}
                  role="region"
                  aria-labelledby={`faq-trigger-${i}`}
                  className={cn(
                    "grid transition-all duration-200",
                    isOpen ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
                  )}
                >
                  <div className="overflow-hidden">
                    <p className="text-muted-foreground px-6 pb-4 text-sm leading-relaxed">
                      {faq.answer}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
