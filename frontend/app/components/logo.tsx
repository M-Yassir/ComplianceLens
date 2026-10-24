import React from "react";

interface LogoProps {
  className?: string;
  size?: number;
}

export default function ComplianceLensLogo({ className = "w-10 h-10", size = 40 }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 512 512"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <defs>
        <linearGradient id="prismGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#0284C7" />
          <stop offset="50%" stopColor="#2563EB" />
          <stop offset="100%" stopColor="#4F46E5" />
        </linearGradient>
      </defs>

      {/* Outer Loupe Rim */}
      <circle
        cx="210"
        cy="210"
        r="140"
        stroke="url(#prismGrad)"
        strokeWidth="44"
        fill="none"
      />

      {/* Tapered Modern Handle */}
      <path
        d="M 315 315 L 435 435"
        stroke="url(#prismGrad)"
        strokeWidth="48"
        strokeLinecap="round"
      />

      {/* Refracted Lens Aperture/Focus Ring */}
      <circle
        cx="210"
        cy="210"
        r="80"
        stroke="#38BDF8"
        strokeWidth="16"
        strokeDasharray="360 40"
        fill="none"
      />

      {/* Center AI Focal Core */}
      <circle cx="210" cy="210" r="28" fill="#2563EB" />
      <path
        d="M 210 170 L 210 250 M 170 210 L 250 210"
        stroke="white"
        strokeWidth="10"
        strokeLinecap="round"
      />
    </svg>
  );
}