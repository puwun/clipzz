import Link from "next/link";
import Image from "next/image";
import { FaXTwitter, FaYoutube, FaInstagram, FaLinkedinIn } from "react-icons/fa6";

const footerLinks = {
  Product: [
    { label: "Features", href: "#features" },
    { label: "How It Works", href: "#how-it-works" },
    { label: "Pricing", href: "#pricing" },
    { label: "FAQ", href: "#faq" },
  ],
  Company: [
    { label: "About", href: "/about" },
    { label: "Contact", href: "/contact" },
  ],
  Legal: [
    { label: "Privacy Policy", href: "/privacy" },
    { label: "Terms of Service", href: "/terms" },
  ],
};

const socialLinks = [
  { icon: FaXTwitter, href: "#", label: "X (Twitter)" },
  { icon: FaYoutube, href: "#", label: "YouTube" },
  { icon: FaInstagram, href: "#", label: "Instagram" },
  { icon: FaLinkedinIn, href: "#", label: "LinkedIn" },
];

export default function Footer() {
  return (
    <footer className="border-t">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-5">
          {/* Brand */}
          <div className="lg:col-span-2">
            <Link href="/" className="inline-flex items-center gap-2">
              <Image
                src="/clipzz_favicon.jpeg"
                alt="Clipzz Logo"
                className="h-10 w-10 rounded-lg"
                width={40}
                height={40}
              />
              <span className="text-xl font-bold text-primary">Clipzz</span>
            </Link>
            <p className="text-muted-foreground mt-3 max-w-sm text-sm leading-relaxed">
              AI-powered podcast clip generator. Turn long-form podcasts into
              viral-ready vertical clips for TikTok, Reels, and Shorts.
            </p>
            <div className="mt-4 flex gap-3">
              {socialLinks.map((s) => {
                const Icon = s.icon;
                return (
                  <a
                    key={s.label}
                    href={s.href}
                    aria-label={s.label}
                    className="text-muted-foreground hover:text-foreground flex h-9 w-9 items-center justify-center rounded-lg border transition-colors hover:bg-accent"
                  >
                    <Icon className="size-4" />
                  </a>
                );
              })}
            </div>
          </div>

          {/* Link columns */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h3 className="text-foreground mb-3 text-sm font-semibold">
                {category}
              </h3>
              <ul className="space-y-2.5">
                {links.map((link) => (
                  <li key={link.label}>
                    {link.href.startsWith("#") ? (
                      <a
                        href={link.href}
                        className="text-muted-foreground hover:text-foreground text-sm transition-colors"
                      >
                        {link.label}
                      </a>
                    ) : (
                      <Link
                        href={link.href}
                        className="text-muted-foreground hover:text-foreground text-sm transition-colors"
                      >
                        {link.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="text-muted-foreground mt-10 flex flex-col items-center justify-between gap-4 border-t pt-6 text-sm sm:flex-row">
          <p>© {new Date().getFullYear()} Clipzz. All rights reserved.</p>
          <p>
            Made with ❤️ for podcasters everywhere
          </p>
        </div>
      </div>
    </footer>
  );
}
