import { motion, AnimatePresence } from "framer-motion"
import { Mic, Power, Loader2 } from "lucide-react"
import type { VoiceState } from "@/hooks/useVoiceSession"

interface PowerButtonProps {
  state: VoiceState
  onToggle: () => void
}

const ringColor: Record<VoiceState, string> = {
  disconnected: "rgba(168,85,247,0.35)",
  connecting: "rgba(217,70,239,0.45)",
  listening: "rgba(236,72,153,0.55)",
  speaking: "rgba(236,72,153,0.85)",
}

export function PowerButton({ state, onToggle }: PowerButtonProps) {
  const active = state !== "disconnected"
  const isSpeaking = state === "speaking"
  const isListening = state === "listening"
  const isConnecting = state === "connecting"

  return (
    <div className="relative flex items-center justify-center">
      {/* Outer animated rings */}
      <AnimatePresence>
        {active &&
          [0, 1, 2].map((i) => (
            <motion.span
              key={i}
              className="absolute rounded-full"
              style={{
                width: 200,
                height: 200,
                border: `2px solid ${ringColor[state]}`,
              }}
              initial={{ scale: 1, opacity: 0.6 }}
              animate={{
                scale: isSpeaking ? [1, 2.1, 1] : [1, 1.7, 1],
                opacity: [0.55, 0, 0.55],
              }}
              exit={{ opacity: 0 }}
              transition={{
                duration: isSpeaking ? 1.4 : 2.6,
                repeat: Infinity,
                delay: i * (isSpeaking ? 0.45 : 0.8),
                ease: "easeOut",
              }}
            />
          ))}
      </AnimatePresence>

      {/* Glow halo */}
      <motion.div
        className="absolute rounded-full blur-2xl"
        style={{
          width: 240,
          height: 240,
          background: isSpeaking
            ? "radial-gradient(circle, rgba(236,72,153,0.7), transparent 70%)"
            : isListening
              ? "radial-gradient(circle, rgba(217,70,239,0.5), transparent 70%)"
              : "radial-gradient(circle, rgba(168,85,247,0.35), transparent 70%)",
        }}
        animate={{
          scale: isSpeaking ? [1, 1.25, 1] : active ? [1, 1.08, 1] : 1,
          opacity: active ? [0.7, 1, 0.7] : 0.4,
        }}
        transition={{
          duration: isSpeaking ? 0.9 : 2.4,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Main button */}
      <motion.button
        onClick={onToggle}
        aria-label={active ? "Stop conversation" : "Start conversation"}
        className="relative z-10 flex h-44 w-44 cursor-pointer items-center justify-center rounded-full outline-none focus-visible:ring-4 focus-visible:ring-pink-400/60"
        style={{
          background: active
            ? "linear-gradient(145deg, #ec4899, #a855f7)"
            : "linear-gradient(145deg, #2a1245, #1a0a2e)",
          boxShadow: active
            ? "0 0 60px rgba(236,72,153,0.6), inset 0 2px 12px rgba(255,255,255,0.25)"
            : "0 0 30px rgba(168,85,247,0.25), inset 0 2px 10px rgba(255,255,255,0.08)",
        }}
        whileHover={{ scale: 1.04 }}
        whileTap={{ scale: 0.95 }}
        animate={isSpeaking ? { scale: [1, 1.05, 1] } : { scale: 1 }}
        transition={isSpeaking ? { duration: 0.6, repeat: Infinity } : { duration: 0.2 }}
      >
        <div className="flex flex-col items-center gap-2 text-white">
          {isConnecting ? (
            <Loader2 className="h-12 w-12 animate-spin" strokeWidth={1.5} />
          ) : isListening || isSpeaking ? (
            <Mic className="h-12 w-12" strokeWidth={1.5} />
          ) : (
            <Power className="h-12 w-12" strokeWidth={1.5} />
          )}
        </div>
      </motion.button>
    </div>
  )
}
