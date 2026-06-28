import { Analytics } from "@vercel/analytics/react";
import "./globals.css";

export const metadata = {
  title: "Pitwall | F1 Predictive Analytics",
  description: "AI-driven Formula 1 race predictions using custom Masked Autoencoders and live telemetry.",
  openGraph: {
    title: "Pitwall | F1 Predictive Analytics",
    description: "AI-driven Formula 1 race predictions using custom Masked Autoencoders and live telemetry.",
    url: "https://pitwall-f1-six.vercel.app/",
    siteName: "Pitwall",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Pitwall F1 Dashboard Preview",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Pitwall | F1 Predictive Analytics",
    description: "AI-driven Formula 1 race predictions using custom Masked Autoencoders and live telemetry.",
    images: ["/og-image.png"],
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
      </head>
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
