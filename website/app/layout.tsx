import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const siteUrl = new URL(`${protocol}://${host}`);
  const socialImage = new URL("/og.png", siteUrl).toString();

  return {
    metadataBase: siteUrl,
    title: "ViroBind — Drug–virus interaction prediction",
    description: "A deep-learning framework for CPI prediction and antiviral compound prioritization.",
    openGraph: {
      title: "ViroBind — Drug–virus interaction prediction",
      description: "From compound libraries to focused antiviral hypotheses.",
      type: "website",
      images: [{ url: socialImage, width: 1731, height: 909, alt: "ViroBind drug–virus interaction prediction" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "ViroBind — Drug–virus interaction prediction",
      description: "From compound libraries to focused antiviral hypotheses.",
      images: [socialImage],
    },
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
