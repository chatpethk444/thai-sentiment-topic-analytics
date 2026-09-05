import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Wisesight Sentiment Dashboard",
  description: "Wisesight Thai sentiment and topic analytics dashboard",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
