"use client";

import { createContext, useContext, useState, useEffect } from "react";
import { themes } from "~/lib/themes";

type ThemeContextType = {
  setRandomTheme: () => void;
  currentThemeName: string;
  currentFont: string;
};

const ThemeContext = createContext<ThemeContextType>({
  setRandomTheme: () => {
      console.warn("setRandomTheme is not implemented");
  },
  currentThemeName: "default",
  currentFont: "sans-serif",
});

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const [currentThemeName, setCurrentThemeName] = useState("default");
  const [currentFont, setCurrentFont] = useState("sans-serif");

  useEffect(() => {
    // Retrieve theme from localStorage on the client side
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme) {
    //   console.log("Stored theme found:", storedTheme);
      setCurrentThemeName(storedTheme);
    }
  }, []);

  useEffect(() => {
    const theme = themes.find((t) => t.name === currentThemeName);
    if (theme) {
      // Apply all theme variables
      for (const [key, value] of Object.entries(theme.variables || {})) {
        // console.log("Setting theme variable:", key, value);
        document.documentElement.style.setProperty(key, value?.toString() || "");
      }
      // Set the font
      const font = theme.font || "sans-serif";
      setCurrentFont(font);
      document.documentElement.style.setProperty("--font-sans", font);
    //   console.log("Setting font:", font);
    }
  }, [currentThemeName]);

  const setRandomTheme = () => {
    const theme = themes[Math.floor(Math.random() * themes.length)];
    if (!theme) return;

    setCurrentThemeName(theme.name);
    localStorage.setItem("theme", theme.name);
  };

  return (
    <ThemeContext.Provider value={{ setRandomTheme, currentThemeName, currentFont }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);