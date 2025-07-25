"use client";

import { ThemeProvider } from "~/context/ThemeContext";

export default function ThemeClientProvider({
  children,
}: { children: React.ReactNode }) {
  return <ThemeProvider>{children}</ThemeProvider>;
}