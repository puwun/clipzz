import { type Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description:
    "Learn how Clipzz collects, uses, and protects your personal information.",
};

export default function PrivacyPage() {
  return (
    <div className="pt-24">
      <section className="py-16">
        <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
          <h1 className="text-foreground text-4xl font-bold tracking-tight sm:text-5xl">
            Privacy Policy
          </h1>
          <p className="text-muted-foreground mt-4 text-sm">
            Last updated: April 10, 2026
          </p>

          <div className="prose-sm mt-10 space-y-8">
            <section>
              <h2 className="text-foreground text-xl font-semibold">
                1. Introduction
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                Clipzz (&ldquo;we&rdquo;, &ldquo;our&rdquo;, or &ldquo;us&rdquo;) is committed to protecting
                your privacy. This Privacy Policy explains how we collect, use,
                disclose, and safeguard your information when you use our
                AI-powered podcast clip generator service.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                2. Information We Collect
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                We collect information you provide directly:
              </p>
              <ul className="text-muted-foreground mt-2 list-disc space-y-1.5 pl-5 text-sm">
                <li>
                  <strong className="text-foreground">Account information:</strong> name, email
                  address, and profile picture (via Google OAuth or email
                  registration)
                </li>
                <li>
                  <strong className="text-foreground">Uploaded content:</strong> podcast video files
                  you upload for processing
                </li>
                <li>
                  <strong className="text-foreground">Payment information:</strong> processed
                  securely through Razorpay (India) or Stripe (international);
                  we do not store your card details
                </li>
                <li>
                  <strong className="text-foreground">Usage data:</strong> how you interact with the
                  service, including features used and processing history
                </li>
              </ul>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                3. How We Use Your Information
              </h2>
              <ul className="text-muted-foreground mt-2 list-disc space-y-1.5 pl-5 text-sm">
                <li>To provide and maintain the Clipzz service</li>
                <li>To process your podcast videos and generate clips</li>
                <li>To manage your account and credit balance</li>
                <li>To process payments securely</li>
                <li>To send important service notifications</li>
                <li>To improve our AI models and service quality</li>
              </ul>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                4. Data Storage & Security
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                Your uploaded videos and generated clips are stored on Amazon Web
                Services (AWS) S3 with server-side encryption. All data
                transmission is encrypted using TLS/SSL. We implement
                industry-standard security measures to protect your data from
                unauthorized access, alteration, or destruction.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                5. Third-Party Services
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                We use the following third-party services:
              </p>
              <ul className="text-muted-foreground mt-2 list-disc space-y-1.5 pl-5 text-sm">
                <li>
                  <strong className="text-foreground">Google OAuth:</strong> for secure
                  authentication
                </li>
                <li>
                  <strong className="text-foreground">Razorpay & Stripe:</strong> for payment
                  processing
                </li>
                <li>
                  <strong className="text-foreground">AWS S3:</strong> for secure file storage
                </li>
                <li>
                  <strong className="text-foreground">Google Gemini:</strong> for AI-powered moment
                  detection (transcription data only)
                </li>
              </ul>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                6. Data Retention
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                We retain your account data as long as your account is active.
                Uploaded videos and generated clips can be deleted at any time
                from your dashboard. Upon account deletion, all associated data
                is permanently removed within 30 days.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                7. Your Rights
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                You have the right to access, correct, or delete your personal
                data. You may export your data or request account deletion by
                contacting us at{" "}
                <a
                  href="mailto:support@clipzz.com"
                  className="text-primary underline underline-offset-2"
                >
                  support@clipzz.com
                </a>
                .
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                8. Contact Us
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                If you have questions about this Privacy Policy, contact us at{" "}
                <a
                  href="mailto:support@clipzz.com"
                  className="text-primary underline underline-offset-2"
                >
                  support@clipzz.com
                </a>
                .
              </p>
            </section>
          </div>
        </div>
      </section>
    </div>
  );
}
