"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import {
  Fish, ShieldCheck, Zap, BarChart2, Eye, Layers, Brain,
  ChevronDown, Sparkles, Target, Waves,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface PredictResponse {
  status: string;
  prediction: {
    label: string;
    confidence: number;
    decision: string;
  };
  detection: {
    bbox: [number, number, number, number];
    mask_coverage: number;
  };
  images: { original: string; roi: string; gradcam: string };
  metadata: {
    timestamp: string;
    processing_time_ms: number;
    model_versions: { segmentor: string; classifier: string };
  };
  explanation: { focus_areas: string[]; note: string; llm_analysis: string };
}

// ── Constants ─────────────────────────────────────────────────────────────────

const API_URL = "http://localhost:8000/predict";

const LOADING_STEPS = [
  { label: "Segmenting fish region", icon: "✂️" },
  { label: "Classifying freshness",  icon: "🧠" },
  { label: "Generating GradCAM map", icon: "🔥" },
];

const TIPS = [
  "U²-Net segments the fish at pixel level — no bounding boxes needed.",
  "Fresh fish have clear, bright eyes. Cloudiness signals spoilage.",
  "Gill color transitions from vivid red to brown as fish age.",
  "GradCAM highlights the exact anatomy the classifier focused on.",
  "EfficientNetV2S achieves top accuracy with far fewer parameters.",
];

// ── Framer Motion variants ────────────────────────────────────────────────────

const fadeUp = {
  hidden: { opacity: 0, y: 32 },
  visible: (i = 0) => ({
    opacity: 1, y: 0,
    transition: { duration: 0.65, ease: "easeOut" as const, delay: i * 0.12 },
  }),
};

const fadeIn = {
  hidden: { opacity: 0 },
  visible: (i = 0) => ({
    opacity: 1,
    transition: { duration: 0.5, ease: "easeOut" as const, delay: i * 0.1 },
  }),
};

// ── Custom hook: animated count-up ───────────────────────────────────────────

function useCountUp(target: number, active: boolean): number {
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!active) { setVal(0); return; }
    let start: number | null = null;
    const dur = 1200;
    const tick = (ts: number) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(Math.round(eased * target));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target, active]);
  return val;
}

// ── SVG circular confidence gauge ─────────────────────────────────────────────

function CircularGauge({ pct, fresh }: { pct: number; fresh: boolean }) {
  const r    = 54;
  const circ = 2 * Math.PI * r;
  const fill = (pct / 100) * circ;
  const color = fresh ? "#10b981" : "#ef4444";
  return (
    <div className="relative w-40 h-40 flex items-center justify-center flex-shrink-0">
      <svg className="absolute inset-0 -rotate-90" width="160" height="160">
        <circle cx="80" cy="80" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="12" />
        <circle cx="80" cy="80" r={r} fill="none"
          stroke={color} strokeWidth="12" strokeLinecap="round"
          strokeDasharray={`${fill} ${circ - fill}`}
          style={{ transition: "stroke-dasharray 1.2s cubic-bezier(0.22,1,0.36,1)" }}
        />
      </svg>
      <div className="flex flex-col items-center z-10 pointer-events-none">
        <span className="text-4xl font-black tabular-nums leading-none" style={{ color }}>
          {pct}%
        </span>
        <span className="text-[10px] uppercase tracking-widest text-slate-500 mt-1">
          confidence
        </span>
      </div>
    </div>
  );
}

// ── Lightbox ──────────────────────────────────────────────────────────────────

function Lightbox({ src, label, onClose }: { src: string; label: string; onClose: () => void }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[100] flex flex-col items-center justify-center animate-reveal-scale"
      style={{ background: "rgba(3,7,18,0.92)", backdropFilter: "blur(24px)" }}
      onClick={onClose}
    >
      <p className="text-xs text-slate-400 mb-3 uppercase tracking-widest">{label}</p>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src} alt={label}
        className="max-h-[80vh] max-w-[90vw] rounded-2xl shadow-2xl object-contain border border-white/10"
        onClick={(e) => e.stopPropagation()}
      />
      <p className="text-xs text-slate-600 mt-3">Click anywhere or press Esc to close</p>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// ── LANDING PAGE COMPONENTS ───────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

// ── Animated ocean background ─────────────────────────────────────────────────

// ── Fish SVG path (reused across background) ─────────────────────────────────
const FISH_PATH = "M178 50 C158 18 115 8 75 28 C38 46 18 50 0 50 C18 50 38 54 75 72 C115 92 158 82 178 50Z";
const FISH_TAIL = "M0 50 L-22 26 L-10 50 L-22 74 Z";

const BG_FISH = [
  { top: "12%",  dur: "32s", delay: "0s",   size: 300, opacity: 0.18, dir: "l", color: "#0A9396" },
  { top: "38%",  dur: "24s", delay: "9s",   size: 190, opacity: 0.14, dir: "r", color: "#94D2BD" },
  { top: "62%",  dur: "38s", delay: "3s",   size: 380, opacity: 0.11, dir: "l", color: "#005F73" },
  { top: "78%",  dur: "28s", delay: "17s",  size: 150, opacity: 0.16, dir: "r", color: "#0A9396" },
  { top: "26%",  dur: "45s", delay: "22s",  size: 240, opacity: 0.09, dir: "l", color: "#94D2BD" },
  { top: "55%",  dur: "20s", delay: "12s",  size: 130, opacity: 0.13, dir: "r", color: "#0A9396" },
];

const RIPPLES = [
  { left: "12%", top: "22%", delay: "0s" },
  { left: "68%", top: "48%", delay: "1.4s" },
  { left: "38%", top: "72%", delay: "2.8s" },
  { left: "82%", top: "18%", delay: "0.7s" },
  { left: "52%", top: "60%", delay: "3.5s" },
];

function OceanBackground() {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none" aria-hidden>
      {/* Base gradient */}
      <div className="absolute inset-0" style={{
        background: "radial-gradient(ellipse 120% 80% at 50% -10%, #003344 0%, #001219 55%, #000a10 100%)",
      }} />

      {/* Drifting bioluminescent orbs */}
      <div className="absolute animate-drift" style={{
        top: "15%", left: "10%", width: 600, height: 600,
        borderRadius: "50%",
        background: "radial-gradient(circle, rgba(10,147,150,0.12) 0%, transparent 65%)",
        filter: "blur(40px)",
      }} />
      <div className="absolute animate-drift2" style={{
        top: "40%", right: "5%", width: 800, height: 800,
        borderRadius: "50%",
        background: "radial-gradient(circle, rgba(0,95,115,0.1) 0%, transparent 65%)",
        filter: "blur(60px)",
      }} />
      <div className="absolute animate-drift3" style={{
        bottom: "10%", left: "30%", width: 500, height: 500,
        borderRadius: "50%",
        background: "radial-gradient(circle, rgba(148,210,189,0.07) 0%, transparent 65%)",
        filter: "blur(50px)",
      }} />

      {/* ── Swimming fish ───────────────────────────────────────────────────── */}
      {BG_FISH.map((f, i) => (
        <div key={i} className="absolute" style={{
          top: f.top,
          animation: `fish-swim-${f.dir} ${f.dur} linear ${f.delay} infinite`,
        }}>
          <svg
            width={f.size}
            viewBox="0 0 200 100"
            style={{
              opacity: f.opacity,
              fill: f.color,
              transform: f.dir === "r" ? "scaleX(-1)" : undefined,
              filter: `drop-shadow(0 0 8px ${f.color}88)`,
            }}
          >
            <path d={FISH_TAIL} />
            <path d={FISH_PATH} />
            {/* Dorsal fin */}
            <path d="M120 28 C130 14 148 14 140 30 Z" fill="rgba(255,255,255,0.18)" />
            {/* Pectoral fin */}
            <path d="M100 55 C110 68 125 65 118 52 Z" fill="rgba(255,255,255,0.12)" />
            {/* Eye */}
            <circle cx="158" cy="42" r="6" fill="rgba(255,255,255,0.55)" />
            <circle cx="159" cy="41" r="2.5" fill="rgba(0,0,0,0.6)" />
            {/* Scale shimmer lines */}
            <path d="M90 35 Q100 50 90 65" stroke="rgba(255,255,255,0.08)" strokeWidth="1.5" fill="none"/>
            <path d="M110 32 Q122 50 110 68" stroke="rgba(255,255,255,0.06)" strokeWidth="1.5" fill="none"/>
          </svg>
        </div>
      ))}

      {/* ── Water ripple circles ────────────────────────────────────────────── */}
      {RIPPLES.map((r, i) => (
        <div key={i} className="absolute" style={{ left: r.left, top: r.top }}>
          {[0, 0.9, 1.8].map((extra, j) => (
            <div key={j} className="absolute rounded-full border" style={{
              width: 80, height: 80,
              marginLeft: -40, marginTop: -40,
              borderColor: "rgba(10,147,150,0.25)",
              animation: `ripple-expand 4s ease-out ${parseFloat(r.delay) + extra}s infinite`,
            }} />
          ))}
        </div>
      ))}

      {/* ── Water wave layers ───────────────────────────────────────────────── */}
      {[
        { top: "18%", opacity: 0.045, dur: "10s", color: "#0A9396" },
        { top: "42%", opacity: 0.030, dur: "14s", color: "#005F73" },
        { top: "68%", opacity: 0.035, dur: "8s",  color: "#94D2BD" },
        { top: "88%", opacity: 0.050, dur: "12s", color: "#0A9396" },
      ].map((w, i) => (
        <div key={i} className="absolute w-full overflow-hidden" style={{ top: w.top, height: 60 }}>
          <svg
            style={{
              width: "200%", height: "100%",
              opacity: w.opacity,
              animation: `wave-scroll ${w.dur} linear ${i % 2 === 0 ? "" : "reverse"} infinite`,
            }}
            viewBox="0 0 2880 60" preserveAspectRatio="none"
          >
            <path
              fill={w.color}
              d="M0,30 C180,55 360,5 540,30 C720,55 900,5 1080,30 C1260,55 1440,5 1620,30 C1800,55 1980,5 2160,30 C2340,55 2520,5 2700,30 C2790,42 2850,18 2880,30 L2880,60 L0,60 Z"
            />
          </svg>
        </div>
      ))}

      {/* Subtle god-ray lines */}
      <svg className="absolute inset-0 w-full h-full opacity-[0.04]" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="ray" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0A9396" />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
        </defs>
        {[15, 28, 42, 55, 68, 78].map((x, i) => (
          <rect key={i} x={`${x}%`} y="0" width={i % 2 === 0 ? "1.5%" : "0.8%"} height="100%"
            fill="url(#ray)" />
        ))}
      </svg>

      {/* Fine dot grid */}
      <div className="absolute inset-0" style={{
        backgroundImage: "radial-gradient(circle, rgba(10,147,150,0.06) 1px, transparent 1px)",
        backgroundSize: "32px 32px",
      }} />
    </div>
  );
}

// ── Hero section ──────────────────────────────────────────────────────────────

function HeroSection() {
  const scrollToApp = () => {
    document.getElementById("try-it")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6 text-center overflow-hidden">

      {/* Floating fish silhouette background art */}
      <div className="absolute inset-0 pointer-events-none select-none" aria-hidden>
        {/* Large fish — top right */}
        <svg className="absolute animate-drift3" style={{ top: "14%", right: "6%", width: 380, opacity: 0.18, filter: "drop-shadow(0 0 16px rgba(10,147,150,0.6))" }}
          viewBox="0 0 200 100" fill="rgba(10,147,150,1)">
          <path d={FISH_TAIL} />
          <path d={FISH_PATH} />
          <path d="M120 28 C130 14 148 14 140 30 Z" fill="rgba(255,255,255,0.2)" />
          <path d="M100 55 C110 68 125 65 118 52 Z" fill="rgba(255,255,255,0.14)" />
          <circle cx="158" cy="42" r="6" fill="rgba(255,255,255,0.6)" />
          <circle cx="159" cy="41" r="2.5" fill="rgba(0,0,0,0.5)" />
        </svg>
        {/* Medium fish — bottom left, flipped */}
        <svg className="absolute animate-drift2" style={{ bottom: "22%", left: "4%", width: 240, opacity: 0.15, transform: "scaleX(-1)", filter: "drop-shadow(0 0 10px rgba(148,210,189,0.5))" }}
          viewBox="0 0 200 100" fill="rgba(148,210,189,1)">
          <path d={FISH_TAIL} />
          <path d={FISH_PATH} />
          <path d="M120 28 C130 14 148 14 140 30 Z" fill="rgba(255,255,255,0.18)" />
          <circle cx="158" cy="42" r="5" fill="rgba(255,255,255,0.55)" />
          <circle cx="159" cy="41" r="2" fill="rgba(0,0,0,0.5)" />
        </svg>
        {/* Small fish — mid right */}
        <svg className="absolute animate-drift" style={{ top: "52%", right: "15%", width: 160, opacity: 0.13, filter: "drop-shadow(0 0 8px rgba(0,95,115,0.5))" }}
          viewBox="0 0 200 100" fill="rgba(0,95,115,1)">
          <path d={FISH_TAIL} />
          <path d={FISH_PATH} />
          <circle cx="158" cy="42" r="5" fill="rgba(255,255,255,0.5)" />
        </svg>
      </div>

      {/* Badge */}
      <motion.div
        variants={fadeUp} initial="hidden" animate="visible" custom={0}
        className="mb-8 inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs font-medium"
        style={{ borderColor: "rgba(10,147,150,0.4)", background: "rgba(10,147,150,0.08)", color: "#94D2BD" }}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-[#0A9396] animate-glow-pulse" />
        YOLOv8 · EfficientNetV2S · Grad-CAM · U²-Net
      </motion.div>

      {/* Main headline */}
      <motion.h1
        variants={fadeUp} initial="hidden" animate="visible" custom={1}
        className="text-6xl sm:text-7xl md:text-8xl font-black tracking-tighter leading-[0.92] mb-6"
      >
        <span style={{
          background: "linear-gradient(135deg, #94D2BD 0%, #0A9396 45%, #005F73 100%)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          backgroundClip: "text",
        }}>
          Freshly
        </span>
        <br />
        <span className="text-white">Fishy</span>
      </motion.h1>

      {/* Sub-headline */}
      <motion.p
        variants={fadeUp} initial="hidden" animate="visible" custom={2}
        className="max-w-xl text-lg sm:text-xl text-slate-400 leading-relaxed mb-10"
      >
        AI-Driven Marine Quality Analysis.{" "}
        <span style={{ color: "#94D2BD" }}>Two-stage computer vision</span> detects, segments,
        classifies, and explains fish freshness in under five seconds.
      </motion.p>

      {/* CTA buttons */}
      <motion.div
        variants={fadeUp} initial="hidden" animate="visible" custom={3}
        className="flex flex-col sm:flex-row gap-4 items-center"
      >
        <button
          onClick={scrollToApp}
          className="animate-float group relative flex items-center gap-2.5 rounded-2xl px-8 py-4 text-sm font-bold text-white transition-all duration-300 hover:scale-105 active:scale-95"
          style={{
            background: "linear-gradient(135deg, #005F73, #0A9396)",
            boxShadow: "0 0 40px rgba(10,147,150,0.35), 0 4px 24px rgba(0,0,0,0.4)",
          }}
        >
          <Fish className="w-4 h-4" />
          Analyse a Fish
          <span className="absolute inset-0 rounded-2xl ring-2 ring-[#0A9396]/0 group-hover:ring-[#0A9396]/40 transition-all duration-300" />
        </button>

        <button
          onClick={() => document.getElementById("pipeline")?.scrollIntoView({ behavior: "smooth" })}
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-[#94D2BD] transition-colors"
        >
          How it works
          <ChevronDown className="w-4 h-4" />
        </button>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        variants={fadeIn} initial="hidden" animate="visible" custom={5}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
      >
        <span className="text-[10px] uppercase tracking-widest text-slate-700">Scroll</span>
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
        >
          <ChevronDown className="w-4 h-4 text-slate-700" />
        </motion.div>
      </motion.div>
    </section>
  );
}

// ── Scanner visual section ────────────────────────────────────────────────────

function ScannerSection() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const [heatmapVisible, setHeatmapVisible] = useState(false);

  useEffect(() => {
    if (inView) setTimeout(() => setHeatmapVisible(true), 1200);
  }, [inView]);

  return (
    <section ref={ref} id="scanner" className="relative py-24 px-6 overflow-hidden">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-2 gap-16 items-center">

          {/* Left: text */}
          <div>
            <motion.p
              variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={0}
              className="text-xs uppercase tracking-widest mb-4"
              style={{ color: "#0A9396" }}
            >
              Two-Stage Pipeline
            </motion.p>
            <motion.h2
              variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={1}
              className="text-4xl sm:text-5xl font-black tracking-tight leading-tight mb-6 text-white"
            >
              Detect. Segment.<br />
              <span style={{
                background: "linear-gradient(90deg, #0A9396, #94D2BD)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}>Understand.</span>
            </motion.h2>
            <motion.p
              variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={2}
              className="text-slate-400 leading-relaxed mb-8"
            >
              YOLOv8 isolates the fish with a sub-8 ms bounding box. U²-Net traces the exact body outline
              pixel by pixel. EfficientNetV2S reads freshness from eyes, gills, and skin texture.
              Grad-CAM makes the decision explainable.
            </motion.p>

            {/* Feature bullets */}
            <motion.div
              variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={3}
              className="space-y-3"
            >
              {[
                { icon: Target,     label: "YOLOv8 detection",    sub: "7.4 ms · 95.6% precision",  color: "#0A9396" },
                { icon: Layers,     label: "U²-Net segmentation", sub: "Pixel-accurate fish mask",   color: "#94D2BD" },
                { icon: Brain,      label: "EfficientNetV2S",     sub: "22k-sample fine-tuned CNN",  color: "#005F73" },
                { icon: Eye,        label: "Grad-CAM heatmap",    sub: "Background-suppressed XAI",  color: "#0A9396" },
              ].map(({ icon: Icon, label, sub, color }, i) => (
                <motion.div
                  key={label}
                  variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={4 + i}
                  className="flex items-center gap-4 ocean-glass rounded-xl px-4 py-3"
                >
                  <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: `${color}18`, border: `1px solid ${color}30` }}>
                    <Icon className="w-4 h-4" style={{ color }} />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white">{label}</p>
                    <p className="text-xs text-slate-500">{sub}</p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>

          {/* Right: animated scanner UI */}
          <motion.div
            variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={1}
            className="flex justify-center"
          >
            <div className="relative w-full max-w-sm">

              {/* Outer scanner frame */}
              <div className="relative rounded-2xl overflow-hidden border border-[#0A9396]/20 ocean-glass p-4"
                style={{ boxShadow: "0 0 80px rgba(10,147,150,0.1), 0 0 0 1px rgba(10,147,150,0.15)" }}>

                {/* Header bar */}
                <div className="flex items-center gap-2 mb-3">
                  <div className="flex gap-1">
                    {["#ef4444", "#f59e0b", "#10b981"].map((c) => (
                      <div key={c} className="w-2.5 h-2.5 rounded-full" style={{ background: c }} />
                    ))}
                  </div>
                  <span className="text-[10px] text-slate-600 font-mono ml-2">YOLO Detector v8 · Live Feed</span>
                  <div className="ml-auto flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-[10px] text-red-400">REC</span>
                  </div>
                </div>

                {/* Fish image area */}
                <div className="relative rounded-xl overflow-hidden bg-[#000d12] aspect-[4/3] flex items-center justify-center">

                  {/* Fish SVG art */}
                  <svg viewBox="0 0 280 210" className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                    {/* Water caustic background */}
                    <defs>
                      <radialGradient id="fishbg" cx="50%" cy="40%" r="60%">
                        <stop offset="0%" stopColor="#001a22" />
                        <stop offset="100%" stopColor="#000a10" />
                      </radialGradient>
                      <radialGradient id="heatGrad" cx="50%" cy="50%" r="50%">
                        <stop offset="0%"   stopColor="#ef4444" stopOpacity="0.9" />
                        <stop offset="40%"  stopColor="#f59e0b" stopOpacity="0.7" />
                        <stop offset="70%"  stopColor="#10b981" stopOpacity="0.4" />
                        <stop offset="100%" stopColor="transparent" stopOpacity="0" />
                      </radialGradient>
                      <clipPath id="fishClip">
                        <path d="M220 105 C200 60 150 40 100 65 C55 85 30 105 10 105 C30 105 55 125 100 145 C150 170 200 150 220 105Z M235 80 C245 80 255 90 255 105 C255 120 245 130 235 130 C220 120 220 90 235 80Z" />
                      </clipPath>
                    </defs>
                    <rect width="280" height="210" fill="url(#fishbg)" />

                    {/* Fish body */}
                    <path d="M220 105 C200 60 150 40 100 65 C55 85 30 105 10 105 C30 105 55 125 100 145 C150 170 200 150 220 105Z"
                      fill="#1a4a5a" />
                    {/* Tail */}
                    <path d="M220 105 C235 80 255 90 255 105 C255 120 235 130 220 105Z"
                      fill="#1a4a5a" />
                    {/* Fish scales pattern */}
                    {[[90,80],[110,75],[130,73],[150,75],[170,80],[90,95],[110,92],[130,90],[150,92],[170,95],[100,110],[120,108],[140,108],[160,110],[100,125],[120,123],[140,123]].map(([cx,cy],i) => (
                      <ellipse key={i} cx={cx} cy={cy} rx="10" ry="7" fill="none" stroke="rgba(0,200,180,0.12)" strokeWidth="0.8" />
                    ))}
                    {/* Eye */}
                    <circle cx="195" cy="91" r="7" fill="#0d2a35" stroke="#0A9396" strokeWidth="1.5" />
                    <circle cx="195" cy="91" r="3" fill="#94D2BD" />
                    <circle cx="193" cy="89" r="1.5" fill="white" />

                    {/* Heatmap overlay */}
                    {heatmapVisible && (
                      <g style={{ opacity: 0, animation: "heatmap-reveal 1s ease forwards 0.2s" }}>
                        <rect x="0" y="0" width="280" height="210" fill="url(#heatGrad)" clipPath="url(#fishClip)" />
                      </g>
                    )}
                  </svg>

                  {/* YOLO bounding box overlay */}
                  <div className="absolute animate-bbox-pulse rounded" style={{
                    top: "14%", left: "3%", right: "8%", bottom: "14%",
                    border: "2px solid rgba(0, 255, 200, 0.7)",
                  }}>
                    {/* Corner markers */}
                    {[
                      "top-0 left-0 border-t-2 border-l-2",
                      "top-0 right-0 border-t-2 border-r-2",
                      "bottom-0 left-0 border-b-2 border-l-2",
                      "bottom-0 right-0 border-b-2 border-r-2",
                    ].map((cls, i) => (
                      <div key={i} className={`absolute ${cls} w-3 h-3 border-[#00ffc8]`}
                        style={{ margin: -2 }} />
                    ))}
                    {/* Label badge */}
                    <div className="absolute -top-6 left-0 flex items-center gap-1.5 rounded px-2 py-0.5 text-[10px] font-bold font-mono"
                      style={{ background: "rgba(0,255,200,0.15)", color: "#00ffc8", border: "1px solid rgba(0,255,200,0.3)" }}>
                      fish  0.97
                    </div>
                  </div>

                  {/* Scan line */}
                  <div className="absolute inset-x-0 h-px pointer-events-none animate-scan-v"
                    style={{ background: "linear-gradient(90deg, transparent, rgba(0,255,200,0.6), transparent)" }} />
                </div>

                {/* Status bar */}
                <div className="mt-3 flex items-center gap-3">
                  <div className="flex-1 h-1 rounded-full bg-[#001a22] overflow-hidden">
                    <div className="h-full rounded-full animate-gradient"
                      style={{ width: "72%", background: "linear-gradient(90deg, #005F73, #0A9396, #94D2BD)" }} />
                  </div>
                  <span className="text-[10px] font-mono text-[#0A9396] whitespace-nowrap">
                    FRESH · 91.2%
                  </span>
                </div>
              </div>

              {/* Floating info chips */}
              <motion.div
                animate={{ y: [0, -6, 0] }} transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                className="absolute -top-5 -right-5 rounded-xl px-3 py-2 text-xs font-semibold ocean-glass"
                style={{ color: "#10b981", border: "1px solid rgba(16,185,129,0.25)" }}
              >
                ✓ Auto Approved
              </motion.div>
              <motion.div
                animate={{ y: [0, 6, 0] }} transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                className="absolute -bottom-5 -left-5 rounded-xl px-3 py-2 text-xs font-mono ocean-glass"
                style={{ color: "#0A9396", border: "1px solid rgba(10,147,150,0.25)" }}
              >
                7.4 ms / frame
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

// ── Metrics dashboard ─────────────────────────────────────────────────────────

function MetricsSection() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  const metrics = [
    {
      icon: Fish,
      value: 22000,
      suffix: "+",
      label: "Training images",
      sub: "Fresh · Not Fresh balanced",
      color: "#0A9396",
    },
    {
      icon: ShieldCheck,
      value: 95.6,
      suffix: "%",
      label: "Detection precision",
      sub: "YOLOv8 on held-out test set",
      color: "#94D2BD",
      decimal: true,
    },
    {
      icon: Zap,
      value: 7.4,
      suffix: " ms",
      label: "Per-frame inference",
      sub: "135+ FPS real-time throughput",
      color: "#005F73",
      decimal: true,
    },
    {
      icon: BarChart2,
      value: 93.8,
      suffix: "%",
      label: "mAP@50 score",
      sub: "Industry-grade detection quality",
      color: "#0A9396",
      decimal: true,
    },
    {
      icon: Brain,
      value: 94.378,
      suffix: "%",
      label: "Classifier test accuracy",
      sub: "EfficientNetV2S on held-out test set",
      color: "#94D2BD",
      decimal: true,
      precision: 3,
    },
  ];

  return (
    <section ref={ref} className="relative py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"}
          className="text-center mb-16"
        >
          <p className="text-xs uppercase tracking-widest mb-3" style={{ color: "#0A9396" }}>
            Model performance
          </p>
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight text-white">
            Numbers that matter
          </h2>
        </motion.div>

        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 sm:gap-5">
          {metrics.map(({ icon: Icon, value, suffix, label, sub, color, decimal, precision }, i) => {
            const scale = decimal ? Math.pow(10, precision ?? 1) : 1;
            const intTarget = decimal ? Math.round(value * scale) : value;
            const counted = useCountUp(intTarget, inView);
            const display = decimal ? (counted / scale).toFixed(precision ?? 1) : counted.toLocaleString();
            return (
              <motion.div
                key={label}
                variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={i}
                className="ocean-glass rounded-2xl p-6 flex flex-col gap-4 group hover:scale-[1.03] transition-transform duration-300"
                style={{ boxShadow: `0 0 40px ${color}0a` }}
              >
                <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ background: `${color}15`, border: `1px solid ${color}25` }}>
                  <Icon className="w-5 h-5" style={{ color }} />
                </div>
                <div>
                  <p className="text-3xl sm:text-4xl font-black tabular-nums leading-none text-white mb-1 animate-count-glow"
                    style={{ color }}>
                    {display}{suffix}
                  </p>
                  <p className="text-sm font-semibold text-slate-200 mb-0.5">{label}</p>
                  <p className="text-xs text-slate-500">{sub}</p>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Ticker tape */}
        <div className="mt-12 overflow-hidden rounded-xl border border-white/5 bg-white/[0.02]">
          <div className="flex items-center animate-ticker whitespace-nowrap py-3 gap-8 text-xs text-slate-600 font-mono select-none">
            {Array(2).fill([
              "mAP@50 · 0.938",
              "Precision · 0.956",
              "Recall · 0.875",
              "mAP@50-95 · 0.757",
              "Test Accuracy · 94.38%",
              "AUC · 0.982",
              "F1 · 0.914",
              "Inference · 7.4 ms",
              "Dataset · 22,000 imgs",
              "Classes · 2",
              "Input · 224×224",
              "Backbone · EfficientNetV2S",
              "Detector · YOLOv8m",
            ]).flat().map((item, i) => (
              <span key={i} className="flex items-center gap-2 flex-shrink-0">
                <span className="w-1 h-1 rounded-full bg-[#0A9396]/40" />
                {item}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

// ── Process flow section ──────────────────────────────────────────────────────

function ProcessSection() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  const steps = [
    {
      icon: Target,
      stage: "01",
      label: "Detection",
      tech: "YOLOv8m",
      desc: "High-res 1280px pass isolates the fish with a tight bounding box in under 8 ms.",
      color: "#0A9396",
    },
    {
      icon: Waves,
      stage: "02",
      label: "Segmentation",
      tech: "U²-Net / rembg",
      desc: "Nested U-Net traces the fish silhouette pixel-by-pixel, producing a clean foreground mask.",
      color: "#94D2BD",
    },
    {
      icon: Sparkles,
      stage: "03",
      label: "Enhancement",
      tech: "CLAHE + BG Blur",
      desc: "Contrast-limited adaptive histogram equalisation sharpens the fish; background is blurred, not zeroed.",
      color: "#005F73",
    },
    {
      icon: Brain,
      stage: "04",
      label: "Classification",
      tech: "EfficientNetV2S",
      desc: "Fine-tuned backbone + GELU head delivers [P(fresh), P(not_fresh)] with a calibrated confidence gate.",
      color: "#0A9396",
    },
    {
      icon: Eye,
      stage: "05",
      label: "Explanation",
      tech: "Grad-CAM",
      desc: "Gradients w.r.t. the last Conv2D, masked by the fish silhouette, highlight eyes · gills · skin.",
      color: "#94D2BD",
    },
  ];

  return (
    <section ref={ref} id="pipeline" className="relative py-24 px-6 overflow-hidden">
      {/* Subtle section divider */}
      <div className="absolute top-0 inset-x-0 h-px"
        style={{ background: "linear-gradient(90deg, transparent, rgba(10,147,150,0.2), transparent)" }} />

      <div className="max-w-6xl mx-auto">
        <motion.div
          variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"}
          className="text-center mb-16"
        >
          <p className="text-xs uppercase tracking-widest mb-3" style={{ color: "#0A9396" }}>
            Pipeline
          </p>
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight text-white">
            Five stages.<br />
            <span style={{
              background: "linear-gradient(90deg, #0A9396, #94D2BD)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}>One verdict.</span>
          </h2>
        </motion.div>

        {/* Desktop: horizontal steps */}
        <div className="hidden lg:flex items-start gap-0 relative">

          {/* ── Connector line + traveling light ── */}
          <div className="absolute top-8 left-[calc(10%+2rem)] right-[calc(10%+2rem)] overflow-visible"
            style={{ height: 1 }}>
            {/* Dim base rail */}
            <div className="absolute inset-0" style={{ background: "rgba(10,147,150,0.15)" }} />

            {/* Glowing orb */}
            <div style={{
              position: "absolute",
              top: "50%",
              width: 28,
              height: 28,
              marginTop: -14,
              borderRadius: "50%",
              background: "radial-gradient(circle, #94D2BD 0%, #0A9396 35%, rgba(10,147,150,0.15) 70%, transparent 100%)",
              boxShadow: "0 0 14px 5px rgba(10,147,150,0.65), 0 0 32px 10px rgba(148,210,189,0.25)",
              animation: "light-travel 4.5s linear infinite",
              filter: "blur(0.5px)",
              zIndex: 20,
            }} />

            {/* Soft tail / comet trail */}
            <div style={{
              position: "absolute",
              top: "50%",
              width: 60,
              height: 3,
              marginTop: -1.5,
              borderRadius: 9999,
              background: "linear-gradient(90deg, transparent 0%, rgba(10,147,150,0.4) 100%)",
              animation: "light-travel 4.5s linear infinite",
              animationDelay: "0s",
              filter: "blur(1px)",
              zIndex: 19,
              transform: "translateX(-60px)",
            }} />
          </div>

          {steps.map(({ icon: Icon, stage, label, tech, desc, color }, i) => {
            // Each step bloom fires when the orb reaches it.
            // Orb travels 4.5 s total; steps are evenly spaced across the line.
            // Peak of step-bloom keyframe is at 8 % of the 4.5 s = 0.36 s into each cycle.
            // So: delay = stepTime − 0.36 s, clamped so the first step starts near 0.
            const stepTimes = [0.2, 1.25, 2.25, 3.25, 4.14];
            const bloomDelay = `${(stepTimes[i] - 0.36).toFixed(2)}s`;

            return (
              <motion.div
                key={stage}
                variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={i}
                className="flex-1 flex flex-col items-center text-center px-4"
              >
                {/* Step circle with bloom ring */}
                <div className="relative w-16 h-16 rounded-2xl flex items-center justify-center mb-5 z-10 ocean-glass"
                  style={{
                    border: `1px solid ${color}35`,
                    boxShadow: `0 0 16px ${color}12`,
                    animation: inView ? `step-bloom 4.5s ease-out infinite ${bloomDelay}` : "none",
                  }}>
                  <Icon className="w-6 h-6" style={{ color }} />
                  <span className="absolute -top-2 -right-2 text-[9px] font-black font-mono rounded-full w-5 h-5 flex items-center justify-center"
                    style={{ background: color, color: "#001219" }}>
                    {stage}
                  </span>
                </div>
                <p className="text-sm font-black text-white mb-0.5">{label}</p>
                <p className="text-[10px] font-mono mb-2" style={{ color }}>{tech}</p>
                <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
              </motion.div>
            );
          })}
        </div>

        {/* Mobile: vertical list */}
        <div className="lg:hidden space-y-4">
          {steps.map(({ icon: Icon, stage, label, tech, desc, color }, i) => (
            <motion.div
              key={stage}
              variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={i}
              className="flex gap-4 ocean-glass rounded-2xl p-5"
            >
              <div className="relative w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ border: `1px solid ${color}35`, background: `${color}10` }}>
                <Icon className="w-5 h-5" style={{ color }} />
                <span className="absolute -top-1.5 -right-1.5 text-[9px] font-black font-mono rounded-full w-4 h-4 flex items-center justify-center"
                  style={{ background: color, color: "#001219" }}>
                  {i + 1}
                </span>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <p className="text-sm font-black text-white">{label}</p>
                  <span className="text-[10px] font-mono" style={{ color }}>{tech}</span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── Live demo preview section ─────────────────────────────────────────────────

function DemoPreviewSection() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  const [hovered, setHovered] = useState(false);

  return (
    <section ref={ref} className="relative py-24 px-6 overflow-hidden">
      <div className="absolute top-0 inset-x-0 h-px"
        style={{ background: "linear-gradient(90deg, transparent, rgba(10,147,150,0.2), transparent)" }} />

      <div className="max-w-5xl mx-auto">
        <motion.div
          variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"}
          className="text-center mb-16"
        >
          <p className="text-xs uppercase tracking-widest mb-3" style={{ color: "#0A9396" }}>
            Explainability preview
          </p>
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight text-white mb-4">
            See what the AI sees
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            Hover over the fish below to reveal the Grad-CAM attention heatmap.
            Warm regions drove the freshness decision.
          </p>
        </motion.div>

        <motion.div
          variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={1}
          className="flex justify-center"
        >
          <div className="relative group cursor-crosshair" style={{ width: 480, maxWidth: "100%" }}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
          >
            {/* Fish scene */}
            <div className="relative rounded-2xl overflow-hidden ocean-glass border-[#0A9396]/20"
              style={{ aspectRatio: "4/3", boxShadow: "0 0 80px rgba(10,147,150,0.15)" }}>

              {/* Base: "photo" */}
              <svg viewBox="0 0 480 360" className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  <radialGradient id="bgPhoto" cx="50%" cy="40%" r="70%">
                    <stop offset="0%" stopColor="#002233" />
                    <stop offset="100%" stopColor="#000d15" />
                  </radialGradient>
                  <radialGradient id="hm1" cx="70%" cy="38%" r="20%">
                    <stop offset="0%"   stopColor="#ef4444" stopOpacity="0.95" />
                    <stop offset="60%"  stopColor="#f59e0b" stopOpacity="0.5" />
                    <stop offset="100%" stopColor="transparent" stopOpacity="0" />
                  </radialGradient>
                  <radialGradient id="hm2" cx="45%" cy="42%" r="25%">
                    <stop offset="0%"   stopColor="#f59e0b" stopOpacity="0.7" />
                    <stop offset="70%"  stopColor="#10b981" stopOpacity="0.3" />
                    <stop offset="100%" stopColor="transparent" stopOpacity="0" />
                  </radialGradient>
                  <radialGradient id="hm3" cx="25%" cy="45%" r="20%">
                    <stop offset="0%"   stopColor="#10b981" stopOpacity="0.5" />
                    <stop offset="100%" stopColor="transparent" stopOpacity="0" />
                  </radialGradient>
                  <clipPath id="bigFishClip">
                    <path d="M390 180 C360 100 270 65 180 105 C100 140 55 180 20 180 C55 180 100 220 180 255 C270 295 360 260 390 180Z M415 140 C435 140 455 158 455 180 C455 202 435 220 415 220 C395 208 395 152 415 140Z" />
                  </clipPath>
                </defs>

                <rect width="480" height="360" fill="url(#bgPhoto)" />

                {/* Ice / surface highlights */}
                {[[60,30,50,15],[200,20,80,12],[350,40,60,18],[120,330,100,14],[300,320,70,10]].map(([x,y,w,h],i) => (
                  <ellipse key={i} cx={x} cy={y} rx={w} ry={h} fill="rgba(148,210,189,0.05)" />
                ))}

                {/* Large fish body */}
                <path d="M390 180 C360 100 270 65 180 105 C100 140 55 180 20 180 C55 180 100 220 180 255 C270 295 360 260 390 180Z"
                  fill="#1e5060" />
                <path d="M415 140 C435 140 455 158 455 180 C455 202 435 220 415 220 C395 208 395 152 415 140Z"
                  fill="#1e5060" />

                {/* Scales */}
                {[
                  [160,120],[190,112],[220,108],[250,110],[280,118],[310,130],[
                  160,140],[190,134],[220,130],[250,132],[280,140],[310,150],[
                  165,160],[195,155],[225,152],[255,154],[285,162],[
                  170,180],[200,176],[230,174],[260,176],[
                  175,200],[205,197],[235,196],
                ].map(([cx,cy],i) => (
                  <ellipse key={i} cx={cx} cy={cy} rx="16" ry="11" fill="none"
                    stroke="rgba(0,200,180,0.1)" strokeWidth="1" />
                ))}

                {/* Dorsal fin */}
                <path d="M200 105 C210 75 240 60 270 70 C250 85 230 95 210 108Z"
                  fill="#163d4a" />

                {/* Eye */}
                <circle cx="355" cy="162" r="12" fill="#0a1f28" stroke="#0A9396" strokeWidth="2" />
                <circle cx="355" cy="162" r="6" fill="#94D2BD" />
                <circle cx="352" cy="159" r="2.5" fill="white" />

                {/* Gill line */}
                <path d="M310 140 C315 160 313 180 308 200" fill="none" stroke="rgba(0,200,180,0.2)" strokeWidth="2" />

                {/* Mouth */}
                <path d="M385 172 C388 176 388 184 385 188" fill="none" stroke="rgba(0,200,180,0.25)" strokeWidth="2" />

                {/* Heatmap overlays — shown on hover */}
                <g style={{
                  opacity: hovered ? 1 : 0,
                  transition: "opacity 0.5s ease",
                }} clipPath="url(#bigFishClip)">
                  <rect width="480" height="360" fill="url(#hm1)" />
                  <rect width="480" height="360" fill="url(#hm2)" />
                  <rect width="480" height="360" fill="url(#hm3)" />
                </g>

                {/* Annotation labels — shown on hover */}
                {hovered && (
                  <g style={{ animation: "heatmap-reveal 0.4s ease forwards" }}>
                    {/* Eye annotation */}
                    <line x1="355" y1="162" x2="395" y2="130" stroke="#ef4444" strokeWidth="1" strokeDasharray="3 2" />
                    <rect x="393" y="118" width="72" height="16" rx="4" fill="rgba(239,68,68,0.15)"
                      stroke="rgba(239,68,68,0.4)" strokeWidth="0.8" />
                    <text x="429" y="129" textAnchor="middle" fontSize="8" fill="#ef4444" fontFamily="monospace">
                      Eye clarity
                    </text>
                    {/* Gill annotation */}
                    <line x1="312" y1="170" x2="275" y2="140" stroke="#f59e0b" strokeWidth="1" strokeDasharray="3 2" />
                    <rect x="224" y="128" width="56" height="16" rx="4" fill="rgba(245,158,11,0.15)"
                      stroke="rgba(245,158,11,0.4)" strokeWidth="0.8" />
                    <text x="252" y="139" textAnchor="middle" fontSize="8" fill="#f59e0b" fontFamily="monospace">
                      Gill color
                    </text>
                    {/* Skin annotation */}
                    <line x1="210" y1="160" x2="180" y2="195" stroke="#10b981" strokeWidth="1" strokeDasharray="3 2" />
                    <rect x="130" y="192" width="72" height="16" rx="4" fill="rgba(16,185,129,0.15)"
                      stroke="rgba(16,185,129,0.4)" strokeWidth="0.8" />
                    <text x="166" y="203" textAnchor="middle" fontSize="8" fill="#10b981" fontFamily="monospace">
                      Skin texture
                    </text>
                  </g>
                )}
              </svg>

              {/* Hover instruction */}
              <div className={`absolute bottom-4 left-1/2 -translate-x-1/2 transition-opacity duration-300 ${hovered ? "opacity-0" : "opacity-100"}`}>
                <div className="flex items-center gap-2 rounded-full px-4 py-2 text-xs text-slate-300 ocean-glass">
                  <Eye className="w-3 h-3 text-[#0A9396]" />
                  Hover to reveal heatmap
                </div>
              </div>

              {/* Active indicator */}
              {hovered && (
                <div className="absolute top-4 right-4 flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold"
                  style={{ background: "rgba(16,185,129,0.15)", border: "1px solid rgba(16,185,129,0.3)", color: "#10b981" }}>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  FRESH · 91.2%
                </div>
              )}
            </div>

            {/* Legend */}
            <div className={`mt-4 flex items-center justify-center gap-6 text-xs transition-opacity duration-500 ${hovered ? "opacity-100" : "opacity-0"}`}>
              {[
                { color: "#ef4444", label: "High attention" },
                { color: "#f59e0b", label: "Medium" },
                { color: "#10b981", label: "Low" },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full" style={{ background: color }} />
                  <span style={{ color }}>{label}</span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// ── Section divider / CTA bridge ──────────────────────────────────────────────

function CTABridge() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <section ref={ref} className="relative py-20 px-6 text-center overflow-hidden">
      <div className="absolute top-0 inset-x-0 h-px"
        style={{ background: "linear-gradient(90deg, transparent, rgba(10,147,150,0.2), transparent)" }} />

      {/* Glow */}
      <div className="absolute inset-0 pointer-events-none" aria-hidden>
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] rounded-full"
          style={{ background: "radial-gradient(ellipse, rgba(10,147,150,0.08) 0%, transparent 70%)", filter: "blur(40px)" }} />
      </div>

      <div className="relative max-w-2xl mx-auto">
        <motion.p
          variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={0}
          className="text-xs uppercase tracking-widest mb-4" style={{ color: "#0A9396" }}
        >
          Live inference
        </motion.p>
        <motion.h2
          variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={1}
          className="text-4xl sm:text-5xl font-black tracking-tight text-white mb-6"
        >
          Try it on your own fish
        </motion.h2>
        <motion.p
          variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={2}
          className="text-slate-400 mb-10 leading-relaxed"
        >
          Upload a photo or use your camera. The full pipeline runs locally — detection,
          segmentation, classification, and Grad-CAM — and returns in under five seconds.
        </motion.p>
        <motion.button
          variants={fadeUp} initial="hidden" animate={inView ? "visible" : "hidden"} custom={3}
          onClick={() => document.getElementById("try-it")?.scrollIntoView({ behavior: "smooth" })}
          className="inline-flex items-center gap-2.5 rounded-2xl px-8 py-4 text-sm font-bold text-white transition-all duration-300 hover:scale-105 active:scale-95"
          style={{
            background: "linear-gradient(135deg, #005F73, #0A9396)",
            boxShadow: "0 0 40px rgba(10,147,150,0.3), 0 4px 24px rgba(0,0,0,0.4)",
          }}
        >
          <Fish className="w-4 h-4" />
          Open the Scanner
          <ChevronDown className="w-4 h-4" />
        </motion.button>
      </div>
    </section>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// ── MAIN COMPONENT ────────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

export default function Home() {
  const videoRef     = useRef<HTMLVideoElement>(null);
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [tab,          setTab]          = useState<"camera" | "upload">("upload");
  const [camActive,    setCamActive]    = useState(false);
  const [result,       setResult]       = useState<PredictResponse | null>(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState<string | null>(null);
  const [previewSrc,   setPreviewSrc]   = useState<string | null>(null);
  const [loadStep,     setLoadStep]     = useState(0);
  const [tipIdx,       setTipIdx]       = useState(0);
  const [isDragOver,   setIsDragOver]   = useState(false);
  const [lightbox,     setLightbox]     = useState<{ src: string; label: string } | null>(null);
  const [shutterFlash, setShutterFlash] = useState(false);

  // ── Webcam ───────────────────────────────────────────────────────────────

  const startCam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 1280, height: 720 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCamActive(true);
      setError(null);
    } catch {
      setError("Camera access denied or unavailable.");
    }
  }, []);

  const stopCam = useCallback(() => {
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCamActive(false);
  }, []);

  useEffect(() => () => stopCam(), [stopCam]);

  const captureFrame = useCallback((): string | null => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")!.drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.92);
  }, []);

  // ── File reading ─────────────────────────────────────────────────────────

  const readFile = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload  = () => resolve(r.result as string);
      r.onerror = reject;
      r.readAsDataURL(file);
    });

  // ── API call ─────────────────────────────────────────────────────────────

  const analyse = useCallback(async (dataUrl: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setPreviewSrc(dataUrl);
    setLoadStep(0);
    const timers = [
      setTimeout(() => setLoadStep(1), 600),
      setTimeout(() => setLoadStep(2), 1600),
      setTimeout(() => setLoadStep(3), 2800),
    ];
    const tipTimer = setInterval(() => setTipIdx((i) => (i + 1) % TIPS.length), 2800);
    try {
      const b64 = dataUrl.includes(",") ? dataUrl.split(",")[1] : dataUrl;
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: b64 }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? "Unknown server error");
      }
      setResult(await res.json());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      timers.forEach(clearTimeout);
      clearInterval(tipTimer);
      setLoading(false);
    }
  }, []);

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleCapture = useCallback(() => {
    setShutterFlash(true);
    setTimeout(() => setShutterFlash(false), 400);
    const frame = captureFrame();
    if (frame) analyse(frame);
  }, [captureFrame, analyse]);

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      stopCam();
      analyse(await readFile(file));
      e.target.value = "";
    },
    [stopCam, analyse],
  );

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (!file || !file.type.startsWith("image/")) return;
      stopCam();
      analyse(await readFile(file));
    },
    [stopCam, analyse],
  );

  // Keyboard shortcuts
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.code === "Space" && camActive && !loading) { e.preventDefault(); handleCapture(); }
      if (e.code === "Escape") setLightbox(null);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [camActive, loading, handleCapture]);

  // ── Derived ──────────────────────────────────────────────────────────────

  const isFresh     = result?.prediction.label === "Fresh";
  const confPct     = result ? Math.round(result.prediction.confidence * 100) : 0;
  const needsHuman  = confPct < 70;
  const counted     = useCountUp(confPct, !!result && !loading);

  const accentColor  = needsHuman ? "#f59e0b" : isFresh ? "#10b981" : "#ef4444";
  const accentBorder = needsHuman ? "rgba(245,158,11,0.25)" : isFresh ? "rgba(16,185,129,0.25)" : "rgba(239,68,68,0.25)";
  const accentGlow   = needsHuman ? "rgba(245,158,11,0.10)" : isFresh ? "rgba(16,185,129,0.12)" : "rgba(239,68,68,0.12)";

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="relative min-h-screen">

      {/* Fixed animated background */}
      <OceanBackground />

      {/* ── LANDING SECTIONS ───────────────────────────────────────────────── */}
      <HeroSection />
      <ScannerSection />
      <MetricsSection />
      <ProcessSection />
      <DemoPreviewSection />
      <CTABridge />

      {/* ── APP SECTION ────────────────────────────────────────────────────── */}
      <section id="try-it" className="relative">
        <div className="absolute top-0 inset-x-0 h-px"
          style={{ background: "linear-gradient(90deg, transparent, rgba(10,147,150,0.4), transparent)" }} />

        <div
          className="h-screen flex flex-col overflow-hidden dot-grid"
          style={{ background: "radial-gradient(ellipse 80% 60% at 15% 0%, #0d1f3c 0%, #030712 65%)" }}
        >
          {/* ── Header ── */}
          <header className="flex-shrink-0 flex items-center gap-3 px-5 h-14 border-b border-white/5"
            style={{ background: "rgba(3,7,18,0.7)", backdropFilter: "blur(20px)" }}>
            <span className="text-xl animate-float inline-block select-none">🐟</span>
            <span className="font-bold text-sm tracking-tight">FreshlyFishy</span>
            <span className="text-[10px] text-slate-600 border border-white/5 rounded-full px-2 py-0.5 ml-1">
              AI Freshness Analysis
            </span>

            <div className="ml-auto flex items-center gap-2">
              {(["U²-Net", "EfficientNetV2S"] as const).map((m, i) => (
                <span key={m} className={`hidden sm:inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] border ${
                  i === 0
                    ? "border-cyan-800/50 bg-cyan-950/40 text-cyan-400"
                    : "border-violet-800/50 bg-violet-950/40 text-violet-400"
                }`}>
                  <span className={`w-1 h-1 rounded-full animate-pulse ${i === 0 ? "bg-cyan-400" : "bg-violet-400"}`} />
                  {m}
                </span>
              ))}

              {result && (
                <button
                  onClick={() => { setResult(null); setPreviewSrc(null); setError(null); }}
                  className="ml-2 flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 px-3 py-1 text-xs text-slate-300 transition-all hover:scale-105 active:scale-95"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>
                  </svg>
                  New Analysis
                </button>
              )}
            </div>
          </header>

          {/* ── Body ── */}
          <div className="flex-1 flex overflow-hidden">

            {/* ══ LEFT PANEL — input ══════════════════════════════════════════ */}
            <aside className="flex-shrink-0 w-full md:w-[400px] flex flex-col border-r border-white/5 overflow-y-auto">

              {/* Tab toggle */}
              <div className="flex gap-1 p-3 border-b border-white/5 flex-shrink-0">
                {(["upload", "camera"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => { setTab(t); if (t !== "camera") stopCam(); }}
                    className={`flex-1 flex items-center justify-center gap-2 rounded-lg py-2 text-xs font-medium transition-all duration-200 ${
                      tab === t
                        ? "bg-white/10 text-white shadow-inner"
                        : "text-slate-500 hover:text-slate-300 hover:bg-white/5"
                    }`}
                  >
                    {t === "camera" ? (
                      <>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>
                        </svg>
                        Camera
                      </>
                    ) : (
                      <>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/>
                          <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                        Upload
                      </>
                    )}
                  </button>
                ))}
              </div>

              {/* ── Camera tab ── */}
              {tab === "camera" && (
                <div className="flex flex-col flex-1 gap-0">
                  <div className="relative bg-black aspect-video overflow-hidden">
                    <video ref={videoRef} className="w-full h-full object-cover" muted playsInline />

                    {camActive && (
                      <div className="absolute inset-0 pointer-events-none">
                        {[["top-3 left-3", "border-t-2 border-l-2"],
                          ["top-3 right-3", "border-t-2 border-r-2"],
                          ["bottom-3 left-3", "border-b-2 border-l-2"],
                          ["bottom-3 right-3", "border-b-2 border-r-2"]].map(([pos, border], i) => (
                          <div key={i} className={`absolute ${pos} w-5 h-5 ${border} border-cyan-400/80`} />
                        ))}
                        <div className="inset-x-0 h-px animate-scanline"
                          style={{ background: "linear-gradient(90deg,transparent,rgba(34,211,238,0.7),transparent)" }} />
                        <div className="absolute top-3 right-3 mt-5">
                          <span className="flex items-center gap-1.5 bg-red-600/90 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider">
                            <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />LIVE
                          </span>
                        </div>
                      </div>
                    )}

                    {shutterFlash && (
                      <div className="absolute inset-0 bg-white pointer-events-none shutter-flash" />
                    )}

                    {!camActive && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-700">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                          <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>
                        </svg>
                        <span className="text-sm">Camera off</span>
                      </div>
                    )}
                  </div>

                  <div className="p-5 flex flex-col items-center gap-4">
                    {!camActive ? (
                      <button
                        onClick={startCam}
                        className="w-full rounded-xl py-3 text-sm font-semibold transition-all duration-200 hover:scale-[1.02] active:scale-95"
                        style={{ background: "linear-gradient(135deg,#0e7490,#0284c7)" }}
                      >
                        Start Camera
                      </button>
                    ) : (
                      <div className="flex flex-col items-center gap-3 w-full">
                        <button
                          onClick={handleCapture}
                          disabled={loading}
                          className="relative w-16 h-16 rounded-full flex items-center justify-center transition-all duration-150 active:scale-90 disabled:opacity-40 group"
                          style={{ background: "rgba(255,255,255,0.08)", border: "3px solid rgba(255,255,255,0.25)" }}
                        >
                          <div className="w-10 h-10 rounded-full bg-white group-hover:bg-slate-200 transition-colors" />
                        </button>
                        <div className="flex items-center gap-1.5 text-slate-600 text-[10px]">
                          <kbd className="border border-white/10 bg-white/5 rounded px-1.5 py-0.5 font-mono">Space</kbd>
                          <span>to capture</span>
                        </div>
                        <button onClick={stopCam}
                          className="w-full rounded-xl border border-white/10 hover:border-white/20 py-2 text-xs text-slate-400 hover:text-slate-200 transition-all">
                          Stop Camera
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* ── Upload tab ── */}
              {tab === "upload" && (
                <div className="flex flex-col flex-1 p-4 gap-4">
                  <div
                    className={`relative flex-1 min-h-[200px] rounded-2xl flex flex-col items-center justify-center gap-4 cursor-pointer transition-all duration-300 select-none ${
                      isDragOver ? "scale-[1.02]" : ""
                    }`}
                    style={{ background: isDragOver ? "rgba(6,182,212,0.06)" : "rgba(15,23,42,0.4)" }}
                    onClick={() => !loading && fileInputRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                    onDragLeave={() => setIsDragOver(false)}
                    onDrop={handleDrop}
                  >
                    <svg className="absolute inset-0 w-full h-full pointer-events-none rounded-2xl overflow-visible">
                      <rect x="1" y="1" width="calc(100% - 2px)" height="calc(100% - 2px)" rx="16" ry="16"
                        fill="none"
                        stroke={isDragOver ? "#06b6d4" : "rgba(255,255,255,0.1)"}
                        strokeWidth={isDragOver ? "1.5" : "1"}
                        strokeDasharray="8 6"
                        className={isDragOver ? "animate-march" : ""}
                        style={{ transition: "stroke 0.2s, stroke-width 0.2s" }}
                      />
                    </svg>

                    {previewSrc ? (
                      /* eslint-disable-next-line @next/next/no-img-element */
                      <img src={previewSrc} alt="preview"
                        className="max-h-48 max-w-full rounded-xl object-contain border border-white/10 shadow-xl" />
                    ) : (
                      <div className={`w-16 h-16 rounded-2xl flex items-center justify-center border border-white/10 transition-all duration-300 ${isDragOver ? "scale-110 border-cyan-500/40" : ""}`}
                        style={{ background: "rgba(30,41,59,0.8)" }}>
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
                          stroke={isDragOver ? "#06b6d4" : "#475569"} strokeWidth="1.5">
                          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                          <polyline points="17 8 12 3 7 8"/>
                          <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                      </div>
                    )}

                    <div className="text-center space-y-1 pointer-events-none">
                      <p className="text-sm font-medium text-slate-300">
                        {isDragOver ? "Drop to analyse" : "Drop image here"}
                      </p>
                      <p className="text-xs text-slate-600">or click to browse · JPEG PNG WEBP</p>
                    </div>
                  </div>

                  {previewSrc && (
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={loading}
                      className="w-full rounded-xl py-2.5 text-sm font-semibold transition-all hover:scale-[1.02] active:scale-95 disabled:opacity-40"
                      style={{ background: "linear-gradient(135deg,#4f46e5,#7c3aed)" }}
                    >
                      Choose different image
                    </button>
                  )}

                  <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="mx-4 mb-4 rounded-xl border border-red-900/50 px-4 py-3 text-red-300 text-xs flex items-center gap-2 animate-reveal-up"
                  style={{ background: "rgba(127,29,29,0.15)" }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0">
                    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  {error}
                </div>
              )}
            </aside>

            {/* ══ RIGHT PANEL — output ══════════════════════════════════════ */}
            <main className="flex-1 overflow-y-auto">

              {/* ── Loading ── */}
              {loading && (
                <div className="h-full flex items-center justify-center p-8">
                  <div className="w-full max-w-md flex flex-col items-center gap-8 animate-reveal-up">
                    <div className="relative w-24 h-24">
                      <div className="absolute inset-0 rounded-full border-2 border-cyan-500/15 animate-ring-pulse" />
                      <div className="absolute inset-3 rounded-full border-2 border-cyan-400/10 animate-ring-pulse" style={{ animationDelay: "0.7s" }} />
                      <div className="absolute inset-0 rounded-full border-t-2 border-r-2 border-cyan-400 animate-spin-slow" />
                      <div className="absolute inset-0 flex items-center justify-center text-3xl">🐟</div>
                    </div>

                    <div className="w-full space-y-3">
                      {LOADING_STEPS.map((step, i) => {
                        const done   = i < loadStep;
                        const active = i === loadStep;
                        return (
                          <div key={i} className={`flex items-center gap-3 transition-all duration-500 ${done ? "opacity-100" : active ? "opacity-70" : "opacity-20"}`}>
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm flex-shrink-0 border transition-all duration-500 ${
                              done   ? "border-emerald-600/60 bg-emerald-900/40 text-emerald-400" :
                              active ? "border-cyan-600/60 bg-cyan-900/30 text-cyan-400" :
                                       "border-slate-700 bg-slate-900/50 text-slate-600"
                            }`}>
                              {done ? "✓" : step.icon}
                            </div>
                            <span className={`text-sm ${done ? "text-emerald-400" : active ? "text-cyan-300" : "text-slate-600"}`}>
                              {step.label}
                              {active && (
                                <span className="inline-flex gap-0.5 ml-2">
                                  {[0,1,2].map((d) => (
                                    <span key={d} className="w-1 h-1 rounded-full bg-cyan-400 animate-bounce inline-block"
                                      style={{ animationDelay: `${d * 0.15}s` }} />
                                  ))}
                                </span>
                              )}
                            </span>
                          </div>
                        );
                      })}
                    </div>

                    <div className="w-full rounded-xl border border-white/5 px-5 py-3.5" style={{ background: "rgba(255,255,255,0.02)" }}>
                      <p className="text-xs text-slate-500 leading-relaxed">
                        <span className="text-slate-400 font-medium">Did you know? </span>
                        {TIPS[tipIdx]}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* ── Empty state ── */}
              {!loading && !result && (
                <div className="h-full flex flex-col items-center justify-center gap-5 p-8 text-center select-none">
                  <div className="text-6xl animate-float">🐠</div>
                  <div className="space-y-1.5">
                    <p className="text-slate-300 font-semibold">No analysis yet</p>
                    <p className="text-slate-600 text-sm max-w-xs">
                      Upload a fish photo or use your camera — results appear here instantly.
                    </p>
                  </div>
                  <div className="flex flex-col gap-2 items-center mt-2">
                    {(["Upload a photo", "Use camera"] as const).map((hint, i) => (
                      <button key={i}
                        onClick={() => { setTab(i === 0 ? "upload" : "camera"); if (i === 1 && !camActive) startCam(); }}
                        className="flex items-center gap-2 text-xs text-slate-500 hover:text-slate-300 transition-colors"
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="9 18 15 12 9 6"/>
                        </svg>
                        {hint}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* ── Results ── */}
              {result && !loading && (
                <div className="p-5 space-y-4 animate-reveal-up">

                  {/* Verdict card */}
                  <div className="rounded-2xl border p-6 relative overflow-hidden glass stagger-1 animate-reveal-up"
                    style={{ borderColor: accentBorder }}>
                    <div className="absolute -top-16 left-1/2 -translate-x-1/2 w-64 h-32 rounded-full blur-3xl pointer-events-none"
                      style={{ background: accentGlow }} />
                    <div className="absolute inset-0 animate-ring-pulse rounded-2xl pointer-events-none"
                      style={{ boxShadow: `inset 0 0 60px ${accentGlow}` }} />

                    <div className="relative flex flex-col sm:flex-row items-center gap-6">
                      <CircularGauge pct={counted} fresh={isFresh!} />

                      <div className="flex-1 text-center sm:text-left space-y-3">
                        <div>
                          <p className="text-xs text-slate-500 uppercase tracking-widest mb-1">Verdict</p>
                          <h2 className="text-5xl font-black tracking-tighter leading-none" style={{ color: accentColor }}>
                            {result.prediction.label.toUpperCase()}
                          </h2>
                        </div>

                        <p className="text-slate-400 text-sm leading-relaxed">
                          {needsHuman
                            ? "Confidence is too low to trust automatically. A human expert should verify this result."
                            : isFresh
                              ? "This fish appears fresh and safe for consumption."
                              : "This fish shows signs of spoilage. Consumption not recommended."}
                        </p>

                        {/* Human intervention banner */}
                        {needsHuman && (
                          <div className="flex items-start gap-3 rounded-xl border border-amber-700/40 bg-amber-900/15 px-4 py-3">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2" className="shrink-0 mt-0.5">
                              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                              <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                            <div>
                              <p className="text-xs font-bold text-amber-300 mb-0.5">Human Intervention Required</p>
                              <p className="text-[11px] text-amber-500/80 leading-snug">
                                Model confidence is below the 70% reliability threshold. Do not act on this result without expert review.
                              </p>
                            </div>
                          </div>
                        )}

                        <div>
                          <div className="flex justify-between text-[10px] text-slate-600 mb-1.5 uppercase tracking-wider">
                            <span>Model confidence</span>
                            <span style={{ color: accentColor }}>{confPct}%</span>
                          </div>
                          <div className="h-1 rounded-full bg-white/5 overflow-hidden">
                            <div className="h-full rounded-full transition-all duration-1000 ease-out"
                              style={{ width: `${confPct}%`, background: `linear-gradient(90deg,${accentColor}80,${accentColor})` }} />
                          </div>
                        </div>

                        <div className="flex flex-wrap gap-2 justify-center sm:justify-start">
                          <span className={`rounded-full px-3 py-1 text-xs font-semibold border ${
                            needsHuman
                              ? "border-amber-700/50 bg-amber-900/30 text-amber-300"
                              : isFresh
                                ? "border-emerald-700/50 bg-emerald-900/30 text-emerald-300"
                                : "border-red-700/50 bg-red-900/30 text-red-300"
                          }`}>
                            {needsHuman ? "Human Intervention Needed" : result.prediction.decision}
                          </span>
                          <span className="rounded-full px-3 py-1 text-xs border border-white/10 bg-white/5 text-slate-400">
                            {result.metadata.processing_time_ms} ms
                          </span>
                          <span className="rounded-full px-3 py-1 text-xs border border-white/10 bg-white/5 text-slate-400">
                            {new Date(result.metadata.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Image trio */}
                  <div className="grid grid-cols-3 gap-3 stagger-2 animate-reveal-up">
                    {([
                      { key: "original" as const, label: "Original",  icon: "📷", desc: "Input"     },
                      { key: "roi"      as const, label: "Fish ROI",  icon: "✂️", desc: "Segmented" },
                      { key: "gradcam"  as const, label: "GradCAM",   icon: "🔥", desc: "Attention" },
                    ]).map(({ key, label, icon, desc }) => (
                      <button
                        key={key}
                        onClick={() => setLightbox({ src: `data:image/jpeg;base64,${result.images[key]}`, label })}
                        className="group rounded-xl overflow-hidden border border-white/5 glass text-left cursor-zoom-in transition-all duration-300 hover:border-white/15 hover:scale-[1.02]"
                      >
                        <div className="relative overflow-hidden h-44 bg-black/40 flex items-center justify-center">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={`data:image/jpeg;base64,${result.images[key]}`} alt={label}
                            className="max-w-full max-h-full object-contain transition-transform duration-500 group-hover:scale-105" />
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <span className="text-xs text-white bg-black/60 rounded-full px-2 py-0.5 backdrop-blur-sm">
                              Click to expand
                            </span>
                          </div>
                        </div>
                        <div className="px-3 py-2 flex items-center gap-1.5">
                          <span className="text-sm">{icon}</span>
                          <div>
                            <p className="text-[11px] font-semibold text-slate-200">{label}</p>
                            <p className="text-[10px] text-slate-600">{desc}</p>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>

                  {/* Stats grid */}
                  <div className="grid sm:grid-cols-2 gap-3 stagger-3 animate-reveal-up">

                    <div className="glass rounded-xl p-4 space-y-3">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-lg bg-cyan-900/40 border border-cyan-800/40 flex items-center justify-center text-xs">✂️</div>
                        <span className="text-xs font-semibold text-slate-200">Segmentation</span>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-slate-500">Fish region</span>
                          <code className="text-slate-300 bg-slate-800/60 px-1.5 py-0.5 rounded text-[10px] font-mono">
                            [{result.detection.bbox.map(Math.round).join(", ")}]
                          </code>
                        </div>
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-slate-500">Mask coverage</span>
                          <span className="text-cyan-300 font-semibold">
                            {(result.detection.mask_coverage * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1 rounded-full bg-white/5 overflow-hidden">
                          <div className="h-full rounded-full bg-cyan-500/60 transition-all duration-700"
                            style={{ width: `${Math.min(result.detection.mask_coverage * 200, 100)}%` }} />
                        </div>
                      </div>
                    </div>

                    <div className="glass rounded-xl p-4 space-y-3">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-lg bg-violet-900/40 border border-violet-800/40 flex items-center justify-center text-xs">⚙️</div>
                        <span className="text-xs font-semibold text-slate-200">Pipeline</span>
                      </div>
                      <div className="space-y-2">
                        {[
                          ["Segmentor",  result.metadata.model_versions.segmentor],
                          ["Classifier", result.metadata.model_versions.classifier],
                          ["Processed",  `${result.metadata.processing_time_ms} ms`],
                        ].map(([k, v]) => (
                          <div key={k} className="flex justify-between items-center text-xs">
                            <span className="text-slate-500">{k}</span>
                            <span className="text-slate-300 text-right max-w-[60%] truncate" title={v}>{v}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="glass rounded-xl p-4 space-y-3 sm:col-span-2 stagger-4 animate-reveal-up">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-lg bg-amber-900/40 border border-amber-800/40 flex items-center justify-center text-xs">💡</div>
                        <span className="text-xs font-semibold text-slate-200">AI Explanation</span>
                      </div>
                      <div>
                        <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">Focus areas</p>
                        <div className="flex flex-wrap gap-1.5">
                          {result.explanation.focus_areas.map((area) => (
                            <span key={area}
                              className="rounded-full px-2.5 py-1 text-[11px] font-medium border border-white/8 text-slate-300"
                              style={{ background: "rgba(30,41,59,0.6)" }}>
                              {area}
                            </span>
                          ))}
                        </div>
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed border-l-2 border-amber-500/40 pl-3">
                        {result.explanation.llm_analysis}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </main>
          </div>

          {/* Hidden canvas */}
          <canvas ref={canvasRef} className="hidden" />
        </div>
      </section>

      {/* Lightbox */}
      {lightbox && (
        <Lightbox src={lightbox.src} label={lightbox.label} onClose={() => setLightbox(null)} />
      )}
    </div>
  );
}
