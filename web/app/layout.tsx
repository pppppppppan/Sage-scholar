import type { Metadata } from "next";
import "./globals.css";
import "katex/dist/katex.min.css";

export const metadata: Metadata = {
  title: "SAGE Scholar",
  description: "Adaptive personal research workspace",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh" className="h-full">
      <body className="h-full overflow-hidden flex flex-col">{children}</body>
    </html>
  );
}
