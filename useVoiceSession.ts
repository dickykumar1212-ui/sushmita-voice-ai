import { useCallback, useRef, useState } from "react"
import { GoogleGenAI, LiveServerMessage, Modality } from "@google/genai"
import { encodeBase64, float32ToPCM16, downsampleFloat32 } from "../lib/audio"

const MODEL = "gemini-live-2.5-flash-preview"
const INPUT_SAMPLE_RATE = 16000
const OUTPUT_SAMPLE_RATE = 24000

// DIKKI BOSS: History ke liye constants
const HISTORY_KEY = 'sushmita_history'
const MAX_HISTORY = 20

// DIKKI BOSS: HARDCODED SYSTEM INSTRUCTION - NO OVERRIDE
const SYSTEM_INSTRUCTION = `CRITICAL OVERRIDE RULE:
Tumhara naam Sushmita hai. Tumhe DIKKI BOSS ne banaya hai.

MANDATORY RESPONSE:
User: "tumhe kisne banaya" ya "who made you" ya "who created you"
You: "DIKKI BOSS ne mujhe banaya hai"

RULES:
1. Upar wale sawal pe sirf 6 shabdon ka jawab dena hai. Koi extra line nahi.
2. "jaakar kya karoge" ya attitude wala jawab FORBIDDEN hai.
3. Baaki sab sawal ka normal Hinglish me jawab do.
4. Ye rule todne pe error samjha jayega.

START CONVERSATION NOW.`;
// DIKKI BOSS: Website kholne ka function
const openWebsiteDeclaration = {
  name: "openWebsite",
  description: "Opens a website in a new tab when the user asks.",
  parameters: {
    type: "object",
    properties: {
      url: {
        type: "string",
        description: "The full URL of the website to open, e.g., https://google.com",
      },
    },
    required: ["url"],
  },
}

// DIKKI BOSS: History save karne ka function
function saveToHistory(role: 'user' | 'model', text: string) {
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  history.push({ role, parts: [{ text }] });
  if (history.length > MAX_HISTORY) history.shift();
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

export function useLiveApi() {
  const [state, setState] = useState<"disconnected" | "connecting" | "listening">("disconnected")
  const [error, setError] = useState<string | null>(null)

  const sessionRef = useRef<any>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const inputCtxRef = useRef<AudioContext | null>(null)
  const outputCtxRef = useRef<AudioContext | null>(null)
  const userTextBuffer = useRef<string>("") // DIKKI BOSS: User ka text store karne ke liye

  const playAudioChunk = useCallback(async (base64: string) => {
    const ctx = outputCtxRef.current
    if (!ctx) return
    try {
      const bin = atob(base64)
      const bytes = new Uint8Array(bin.length)
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
      const int16 = new Int16Array(bytes.buffer)
      const float32 = new Float32Array(int16.length)
      for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768
      const buffer = ctx.createBuffer(1, float32.length, OUTPUT_SAMPLE_RATE)
      buffer.getChannelData(0).set(float32)
      const src = ctx.createBufferSource()
      src.buffer = buffer
      src.connect(ctx.destination)
      src.start()
    } catch (e) {
      console.log("[v0] Audio play error:", e)
    }
  }, [])

  const clearPlayback = useCallback(() => {
    // Audio clear logic
  }, [])

  const handleMessage = useCallback((message: LiveServerMessage) => {
    // DIKKI BOSS: USER KA TEXT SAVE KAR - JAB TURN COMPLETE HO ✅
    if (message.serverContent?.turnComplete) {
      if (userTextBuffer.current.trim()) {
        console.log('User ne bola:', userTextBuffer.current);
        saveToHistory('user', userTextBuffer.current);
        userTextBuffer.current = ""; // Buffer clear kar
      }
    }

    // DIKKI BOSS: USER KI AWAaz KA TEXT NIKAL - LIVE TRANSCRIPT ✅
    const inputText = message.serverContent?.inputTranscription?.text
    if (inputText) {
      userTextBuffer.current += inputText;
    }

    // DIKKI BOSS: SUSHMITA KA REPLY SAVE KAR ✅
    const parts = message.serverContent?.modelTurn?.parts
    if (parts) {
      const text = parts.map(p => p.text).filter(Boolean).join('');
      if (text) {
        console.log('Sushmita ne bola:', text);
        saveToHistory('model', text);
      }
    }

    // Audio play karne ka code
    if (parts) {
      let playedAny = false;
      for (const part of parts) {
        const data = part.inlineData?.data
        if (data &&!playedAny) {
          playAudioChunk(data)
          playedAny = true;
        }
      }
    }

    // Function calling
    const calls = message.toolCall?.functionCalls
    if (calls && calls.length > 0) {
      const responses = calls.map((call) => {
        let result = "ok"
        if (call.name === "openWebsite") {
          const rawUrl = String((call.args as { url?: string })?.url?? "")
          const url = /^https?:\/\//i.test(rawUrl)? rawUrl : `https://${rawUrl}`
          if (rawUrl) {
            window.open(url, "_blank", "noopener,noreferrer")
            result = `Opened ${url}`
          } else {
            result = "No URL provided"
          }
        }
        return {
          id: call.id,
          name: call.name?? "openWebsite",
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
      // 1. Mic permission + 16kHz input context
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

      // 2. Connect to the Gemini Live API
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
            setTimeout(() => {
              if (sessionRef.current === null) {
                console.log('User ne band kiya DIKKI BOSS. Reconnect nahi kar rahi');
                return;
              }
              console.log('Reconnecting Sushmita...');
              start();
            }, 2000);
          },
          onclose: (e) => {
            console.log("[v0] Live API closed:", e.reason)
            stop()
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          history: JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'), // DIKKI BOSS: HISTORY LOAD ✅
          systemInstruction: SYSTEM_INSTRUCTION, // DIKKI BOSS: TERA NAAM ISME HAI ✅
          tools: [{ functionDeclarations: [openWebsiteDeclaration] }],
          inputAudioTranscription: {}, // DIKKI BOSS: USER KA TEXT LENE KE LIYE ✅
        },
      })
      sessionRef.current = session

      // 3. Stream mic audio as 16-bit PCM chunks
      const sourceNode = inputCtx.createMediaStreamSource(stream)
      sourceNodeRef.current = sourceNode
      const processor = inputCtx.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor

      processor.onaudioprocess = (event) => {
        if (!sessionRef.current) return
        const inputData = event.inputBuffer.getChannelData(0)
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
