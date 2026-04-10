"use client";

import { type FormEvent, useState } from "react";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { Mail, MapPin, MessageSquare } from "lucide-react";
import Link from "next/link";

export default function ContactPage() {
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    // In production, this would send data to a server action
    setSubmitted(true);
  };

  return (
    <div className="pt-24">
      <section className="py-16">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="mb-12 text-center">
            <h1 className="text-foreground text-4xl font-bold tracking-tight sm:text-5xl">
              Contact Us
            </h1>
            <p className="text-muted-foreground mx-auto mt-4 max-w-xl text-lg">
              Have a question, feature request, or just want to say hello?
              We&apos;d love to hear from you.
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-5">
            {/* Contact Info */}
            <div className="space-y-6 md:col-span-2">
              <div className="flex items-start gap-3">
                <div className="bg-primary/10 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg">
                  <Mail className="text-primary size-5" />
                </div>
                <div>
                  <h3 className="text-foreground text-sm font-semibold">Email</h3>
                  <p className="text-muted-foreground text-sm">
                    support@clipzz.com
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="bg-primary/10 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg">
                  <MapPin className="text-primary size-5" />
                </div>
                <div>
                  <h3 className="text-foreground text-sm font-semibold">Location</h3>
                  <p className="text-muted-foreground text-sm">
                    Bangalore, India
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="bg-primary/10 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg">
                  <MessageSquare className="text-primary size-5" />
                </div>
                <div>
                  <h3 className="text-foreground text-sm font-semibold">FAQ</h3>
                  <p className="text-muted-foreground text-sm">
                    Check our{" "}
                    <Link
                      href="/#faq"
                      className="text-primary underline underline-offset-2"
                    >
                      FAQ section
                    </Link>{" "}
                    for quick answers.
                  </p>
                </div>
              </div>
            </div>

            {/* Contact Form */}
            <div className="md:col-span-3">
              <Card>
                <CardHeader>
                  <CardTitle>Send a Message</CardTitle>
                  <CardDescription>
                    Fill out the form below and we&apos;ll get back to you within
                    24 hours.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {submitted ? (
                    <div className="flex flex-col items-center gap-3 py-8 text-center">
                      <div className="bg-primary/10 flex h-14 w-14 items-center justify-center rounded-full">
                        <Mail className="text-primary size-7" />
                      </div>
                      <h3 className="text-foreground text-lg font-semibold">
                        Message Sent!
                      </h3>
                      <p className="text-muted-foreground text-sm">
                        Thanks for reaching out. We&apos;ll be in touch soon.
                      </p>
                    </div>
                  ) : (
                    <form onSubmit={handleSubmit} className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="contact-name">Name</Label>
                        <Input
                          id="contact-name"
                          name="name"
                          placeholder="Your name"
                          required
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="contact-email">Email</Label>
                        <Input
                          id="contact-email"
                          name="email"
                          type="email"
                          placeholder="you@example.com"
                          required
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="contact-message">Message</Label>
                        <textarea
                          id="contact-message"
                          name="message"
                          rows={5}
                          placeholder="How can we help?"
                          required
                          className="border-input bg-background text-foreground placeholder:text-muted-foreground flex w-full rounded-md border px-3 py-2 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
                        />
                      </div>
                      <Button type="submit" className="w-full">
                        Send Message
                      </Button>
                    </form>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
