// Audio helpers for streaming PCM to/from the Gemini Live API.

// Encode a Uint8Array of raw bytes to a base64 string.
export function encodeBase64(bytes: Uint8Array): string {
  let binary = ""
  const chunkSize = 0x8000
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize)
    binary += String.fromCharCode.apply(null, chunk as unknown as number[])
  }
  return btoa(binary)
}

// Decode a base64 string into a Uint8Array.
export function decodeBase64(base64: string): Uint8Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes
}

// Convert Float32 PCM samples (-1..1) from the mic into 16-bit little-endian PCM.
export function float32ToPCM16(input: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(input.length * 2)
  const view = new DataView(buffer)
  for (let i = 0; i < input.length; i++) {
    let s = Math.max(-1, Math.min(1, input[i]))
    s = s < 0 ? s * 0x8000 : s * 0x7fff
    view.setInt16(i * 2, s, true)
  }
  return buffer
}

// Linearly downsample Float32 audio from one sample rate to a lower target rate.
// Gemini Live expects 16kHz input; browsers often give 44.1k/48k.
export function downsampleFloat32(
  input: Float32Array,
  inputRate: number,
  targetRate: number,
): Float32Array {
  if (targetRate >= inputRate) return input
  const ratio = inputRate / targetRate
  const newLength = Math.floor(input.length / ratio)
  const result = new Float32Array(newLength)
  let pos = 0
  for (let i = 0; i < newLength; i++) {
    const start = Math.floor(pos)
    const end = Math.min(Math.floor(pos + ratio), input.length)
    let sum = 0
    let count = 0
    for (let j = start; j < end; j++) {
      sum += input[j]
      count++
    }
    result[i] = count > 0 ? sum / count : 0
    pos += ratio
  }
  return result
}

// Convert raw 16-bit PCM (little-endian) into an AudioBuffer for playback.
export function pcm16ToAudioBuffer(
  pcmBytes: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
): AudioBuffer {
  const view = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength)
  const sampleCount = Math.floor(pcmBytes.byteLength / 2)
  const audioBuffer = ctx.createBuffer(1, sampleCount, sampleRate)
  const channel = audioBuffer.getChannelData(0)
  for (let i = 0; i < sampleCount; i++) {
    channel[i] = view.getInt16(i * 2, true) / 0x8000
  }
  return audioBuffer
}
