import { motion } from "framer-motion"
import { useVoiceSession } from "@/hooks/useVoiceSession"
import { PowerButton } from "@/components/PowerButton"
import { StatusLabel } from "@/components/StatusLabel"

export default function App() {
  const { state, error, start, stop } = useVoiceSession()

  const onToggle = () => {
    if (state === "disconnected") {
      void start()
    } else {
      stop()
    }
  }

  return (
    <main className="relative flex min-h-dvh flex-col items-center justify-between overflow-hidden bg-background px-6 py-12">
      {/* Ambient background gradient */}
      <div
        className="pointer-events-none absolute inset-0 -z-10"
        style={{
          background:
            "radial-gradient(circle at 50% 20%, rgba(168,85,247,0.18), transparent 55%), radial-gradient(circle at 50% 90%, rgba(236,72,153,0.16), transparent 55%)",
        }}
      />

      {/* Header */}
      <header className="flex flex-col items-center gap-2 text-center">
        <motion.h1
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-gradient-to-r from-pink-300 via-fuchsia-300 to-purple-300 bg-clip-text text-5xl font-extrabold tracking-tight text-transparent text-balance"
        >
          SUSHMITA
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="max-w-xs text-sm text-purple-300/70 text-pretty"
        >
          Your witty voice companion. Just tap and talk.
        </motion.p>
      </header>

      {/* Center button */}
      <div className="flex flex-col items-center gap-10">
        <PowerButton state={state} onToggle={onToggle} />
        <StatusLabel state={state} />
      </div>

      {/* Footer / error */}
      <footer className="flex h-10 items-center justify-center">
        {error ? (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="rounded-full border border-pink-500/30 bg-pink-500/10 px-4 py-2 text-sm text-pink-200"
          >
            {error}
          </motion.p>
        ) : (
          <p className="text-xs text-purple-400/40">Powered by Gemini Live · Audio only</p>
        )}
      </footer>
    </main>
  )
}
