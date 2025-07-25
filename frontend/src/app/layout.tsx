import "~/styles/globals.css";
import { type Metadata } from "next";
import ThemeClientProvider from "~/components/ThemeClientProvider";
import { Inter, Roboto } from "next/font/google";


const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-inter",
});

const roboto = Roboto({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-roboto",
});


export const metadata: Metadata = {
  title: "Clipzz",
  description: "A platform for video clipping and processing",
  icons: [{ rel: "icon", url: "/android-chrome-512x512.png" }],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${roboto.variable}`}>
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Roboto:wght@400;500;700&family=Architects+Daughter&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <ThemeClientProvider>
          {children}
        </ThemeClientProvider>
      </body>
    </html>
  );
}