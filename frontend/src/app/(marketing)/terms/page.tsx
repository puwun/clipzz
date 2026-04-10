import { type Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service",
  description:
    "Read the Terms of Service for using the Clipzz AI-powered podcast clip generator.",
};

export default function TermsPage() {
  return (
    <div className="pt-24">
      <section className="py-16">
        <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
          <h1 className="text-foreground text-4xl font-bold tracking-tight sm:text-5xl">
            Terms of Service
          </h1>
          <p className="text-muted-foreground mt-4 text-sm">
            Last updated: April 10, 2026
          </p>

          <div className="prose-sm mt-10 space-y-8">
            <section>
              <h2 className="text-foreground text-xl font-semibold">
                1. Acceptance of Terms
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                By accessing or using Clipzz (&ldquo;the Service&rdquo;), you agree to be
                bound by these Terms of Service. If you do not agree to these
                terms, please do not use the Service.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                2. Description of Service
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                Clipzz is an AI-powered platform that processes podcast video
                files to generate short-form vertical clips. The Service
                includes video upload, speech transcription, AI moment detection,
                active speaker detection, vertical cropping, and animated
                subtitle generation.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                3. User Accounts
              </h2>
              <ul className="text-muted-foreground mt-2 list-disc space-y-1.5 pl-5 text-sm">
                <li>
                  You must create an account to use the Service, either via
                  Google OAuth or email/password registration
                </li>
                <li>
                  You are responsible for maintaining the security of your
                  account credentials
                </li>
                <li>
                  You must be at least 18 years old or have parental consent to
                  use the Service
                </li>
                <li>
                  One person or entity may not maintain more than one account
                </li>
              </ul>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                4. Credits & Payments
              </h2>
              <ul className="text-muted-foreground mt-2 list-disc space-y-1.5 pl-5 text-sm">
                <li>
                  The Service operates on a credit-based system where 1 credit =
                  1 minute of podcast processing
                </li>
                <li>New users receive 10 free credits upon registration</li>
                <li>
                  Credit packages are available for purchase: Small (50 credits /
                  ₹830), Medium (150 credits / ₹2,075), and Large (500 credits /
                  ₹5,810)
                </li>
                <li>All credit purchases are one-time payments, not subscriptions</li>
                <li>Credits do not expire</li>
                <li>
                  Payments are processed securely through Razorpay (India) or
                  Stripe (international)
                </li>
              </ul>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                5. Refund Policy
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                Credit purchases are non-refundable once credits have been used.
                If you experience a technical issue that prevents processing, and
                credits are deducted without clip generation, please contact
                support for a credit restoration.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                6. Content Ownership & Rights
              </h2>
              <ul className="text-muted-foreground mt-2 list-disc space-y-1.5 pl-5 text-sm">
                <li>
                  You retain full ownership of all content you upload and clips
                  generated therefrom
                </li>
                <li>
                  You represent that you have the right to upload and process the
                  content you submit
                </li>
                <li>
                  Clipzz does not claim any ownership rights over your content
                </li>
                <li>
                  You grant Clipzz a limited license to process your content
                  solely for the purpose of providing the Service
                </li>
              </ul>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                7. Acceptable Use
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                You agree not to upload content that is illegal, infringes on
                intellectual property rights, contains harmful or malicious
                material, or violates any applicable laws. We reserve the right
                to terminate accounts that violate these terms.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                8. Service Availability
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                We strive to maintain high availability but do not guarantee
                uninterrupted access to the Service. Processing times may vary
                based on server load and video complexity. We are not liable for
                any losses resulting from service interruptions.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                9. Limitation of Liability
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                To the maximum extent permitted by law, Clipzz shall not be
                liable for any indirect, incidental, special, or consequential
                damages arising from your use of the Service. Our total liability
                shall not exceed the amount you have paid to us in the past 12
                months.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                10. Modifications
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                We reserve the right to modify these Terms at any time. Material
                changes will be communicated via email or through the Service.
                Continued use after changes constitutes acceptance of the updated
                terms.
              </p>
            </section>

            <section>
              <h2 className="text-foreground text-xl font-semibold">
                11. Contact
              </h2>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                For questions about these Terms, contact us at{" "}
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
