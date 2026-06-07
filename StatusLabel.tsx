import { motion, AnimatePresence } from "framer-motion"
import type { VoiceState } from "@/hooks/useVoiceSession"

const labels: Record<VoiceState, { title: string; subtitle: string }> = {
  disconnected: { title: "Tap to wake me up", subtitle: "I'm waiting, Boss." },
  connecting: { title: "Connecting...", subtitle: "Getting ready for you." },
  listening: { title: "Listening", subtitle: "Go on, I'm all ears." },
  speaking: { title: "Talking", subtitle: "Hope you're listening, Boss." },
}

export function StatusLabel({ state }: { state: VoiceState }) {
  const { title, subtitle } = labels[state]
  return (
    <div className="flex h-20 flex-col items-center justify-center text-center">
      <AnimatePresence mode="wait">
        <motion.div
          key={state}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25 }}
        >
          <p className="text-xl font-semibold tracking-tight text-pink-100">{title}</p>
          <p className="mt-1 text-sm text-purple-300/70">{subtitle}</p>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
