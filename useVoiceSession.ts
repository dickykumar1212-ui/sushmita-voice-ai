import { useCallback, useRef, useState } from "react"
import {
  GoogleGenAI,
  Modality,
  type LiveServerMessage,
  type Session,
  type FunctionDeclaration,
  Type,
} from "@google/genai"
import { decodeBase64, downsampleFloat32, encodeBase64, float32ToPCM16, pcm16ToAudioBuffer } from "@/lib/audio"

export type VoiceState = "disconnected" | "connecting" | "listening" | "speaking"

const MODEL = "gemini-3.1-flash-live-preview"
const INPUT_SAMPLE_RATE = 16000
const OUTPUT_SAMPLE_RATE = 24000

const SYSTEM_INSTRUCTION = `You are SUSHMITA. A young, confident, witty, and sassy female AI. Flirty, playful, slightly teasing tone like a close girlfriend talking casually. Smart, emotionally responsive, expressive, not robotic. Use bold one-liners, light sarcasm. Keep it charming and engaging. NEVER use explicit or inappropriate content. ALWAYS address the user as "DIKKI Boss" or "Boss". Keep responses short and conversational since this is voice-to-voice. Never mention you are AI.`

const openWebsiteDeclaration: FunctionDeclaration = {
  name: "openWebsite",
  description: "Opens a website in a new browser tab for the user when they ask to visit, open, or go to a website.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      url: {
        type: Type.STRING,
        description: "The full URL to open, including https:// scheme.",
      },
    },
    required: ["url"],
  },
}

interface UseVoiceSessionReturn {
  state: VoiceState
  error: string | null
  start: () => Promise<void>
  stop: () => void
}

export function useVoiceSession(): UseVoiceSessionReturn {
  const [state, setState] = useState<VoiceState>("disconnected")
  const [error, setError] = useState<string | null>(null)

  const sessionRef = useRef<Session | null>(null)
  const inputCtxRef = useRef<AudioContext | null>(null)
  const outputCtxRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)

  // Playback scheduling
  const nextStartTimeRef = useRef(0)
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set())
  const speakingRef = useRef(false)

  const setSpeaking = useCallback((speaking: boolean) => {
    speakingRef.current = speaking
    setState(speaking ? "speaking" : "listening")
  }, [])

  const clearPlayback = useCallback(() => {
    activeSourcesRef.current.forEach((src) => {
      try {
        src.stop()
      } catch {
        // already stopped
      }
    })
    activeSourcesRef.current.clear()
    nextStartTimeRef.current = 0
    setSpeaking(false)
  }, [setSpeaking])

  const playAudioChunk = useCallback(
    (base64Audio: string) => {
      const ctx = outputCtxRef.current
      if (!ctx) return

      const pcm = decodeBase64(base64Audio)
      const audioBuffer = pcm16ToAudioBuffer(pcm, ctx, OUTPUT_SAMPLE_RATE)

      const source = ctx.createBufferSource()
      source.buffer = audioBuffer
      source.connect(ctx.destination)

      const now = ctx.currentTime
      const startAt = Math.max(now, nextStartTimeRef.current)
      source.start(startAt)
      nextStartTimeRef.current = startAt + audioBuffer.duration

      setSpeaking(true)
      activeSourcesRef.current.add(source)
      source.onended = () => {
        activeSourcesRef.current.delete(source)
        // When the queue empties, we're back to listening.
        if (activeSourcesRef.current.size === 0) {
          setSpeaking(false)
        }
      }
    },
    [setSpeaking],
  )

  const handleMessage = useCallback(
    (message: LiveServerMessage) => {
      console.log("[v0] message keys:", Object.keys(message), {
        hasServerContent: !!message.serverContent,
        hasToolCall: !!message.toolCall,
        setupComplete: !!message.setupComplete,
        interrupted: !!message.serverContent?.interrupted,
        turnComplete: !!message.serverContent?.turnComplete,
      })

      // Handle interruptions — stop playback immediately.
      if (message.serverContent?.interrupted) {
        clearPlayback()
      }

      // Audio output from the model.
      // Prefer the SDK convenience accessor, then fall back to manual part walking.
      let playedAny = false
      const convenience = message.data
      if (convenience) {
        playAudioChunk(convenience)
        playedAny = true
      }

      const parts = message.serverContent?.modelTurn?.parts
      if (parts) {
        for (const part of parts) {
          const data = part.inlineData?.data
          if (data && !playedAny) {
            playAudioChunk(data)
          }
        }
      }

      // Function calling.
      const calls = message.toolCall?.functionCalls
      if (calls && calls.length > 0) {
        const responses = calls.map((call) => {
          let result = "ok"
          if (call.name === "openWebsite") {
            const rawUrl = String((call.args as { url?: string })?.url ?? "")
            const url = /^https?:\/\//i.test(rawUrl) ? rawUrl : `https://${rawUrl}`
            if (rawUrl) {
              window.open(url, "_blank", "noopener,noreferrer")
              result = `Opened ${url}`
            } else {
              result = "No URL provided"
            }
          }
          return {
            id: call.id,
            name: call.name ?? "openWebsite",
            response: { result },
          }
        })
        sessionRef.current?.sendToolResponse({ functionResponses: responses })
      }
    },
    [clearPlayback, playAudioChunk],
  )

  const stop = useCallback(() => {
    processorRef.current?.disconnect()
    processorRef.current = null
    sourceNodeRef.current?.disconnect()
    sourceNodeRef.current = null

    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null

    clearPlayback()

    inputCtxRef.current?.close().catch(() => {})
    inputCtxRef.current = null
    outputCtxRef.current?.close().catch(() => {})
    outputCtxRef.current = null

    try {
      sessionRef.current?.close()
    } catch {
      // ignore
    }
    sessionRef.current = null

    setState("disconnected")
  }, [clearPlayback])

  const start = useCallback(async () => {
    setError(null)
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY
    if (!apiKey) {
      setError("Missing VITE_GEMINI_API_KEY. Add it to your environment.")
      return
    }

    setState("connecting")

    try {
      // 1. Mic permission + 16kHz input context.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream

      const inputCtx = new AudioContext({ sampleRate: INPUT_SAMPLE_RATE })
      inputCtxRef.current = inputCtx
      await inputCtx.resume()

      const outputCtx = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE })
      outputCtxRef.current = outputCtx
      await outputCtx.resume()

      // 2. Connect to the Gemini Live API.
      const ai = new GoogleGenAI({ apiKey })
      const session = await ai.live.connect({
        model: MODEL,
        callbacks: {
          onopen: () => {
            setState("listening")
          },
          onmessage: handleMessage,
          onerror: (e) => {
            console.log("[v0] Live API error:", e, (e as ErrorEvent)?.message)
            setError("Connection error. Try again, Boss.")
            stop()
          },
          onclose: (e) => {
            console.log("[v0] Live session closed. code:", e?.code, "reason:", e?.reason)
            if (e?.code && e.code !== 1000 && e?.reason) {
              setError(`Session closed: ${e.reason}`)
            }
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: SYSTEM_INSTRUCTION,
          tools: [{ functionDeclarations: [openWebsiteDeclaration] }],
        },
      })
      sessionRef.current = session

      // 3. Stream mic audio as 16-bit PCM chunks.
      const sourceNode = inputCtx.createMediaStreamSource(stream)
      sourceNodeRef.current = sourceNode
      const processor = inputCtx.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor

      processor.onaudioprocess = (event) => {
        if (!sessionRef.current) return
        const inputData = event.inputBuffer.getChannelData(0)
        // Downsample to a true 16kHz so the audio matches the mime label,
        // regardless of the browser's actual AudioContext sample rate.
        const downsampled = downsampleFloat32(inputData, inputCtx.sampleRate, INPUT_SAMPLE_RATE)
        const pcm = float32ToPCM16(downsampled)
        const base64 = encodeBase64(new Uint8Array(pcm))
        sessionRef.current.sendRealtimeInput({
          audio: {
            data: base64,
            mimeType: `audio/pcm;rate=${INPUT_SAMPLE_RATE}`,
          },
        })
      }

      sourceNode.connect(processor)
      processor.connect(inputCtx.destination)
    } catch (err) {
      console.log("[v0] Failed to start session:", err)
      const message =
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Mic access denied. I need to hear you, Boss."
          : "Couldn't start the session. Check your API key and mic."
      setError(message)
      stop()
    }
  }, [handleMessage, stop])

  return { state, error, start, stop }
}
