import { onUnmounted } from 'vue'

/**
 * Web Audio API 音频播放器
 * 接收后端对齐 batch 中嵌入的 PCM 切片，用 AudioContext 精确调度播放。
 * 音频与视频帧共享同一 globalSec 时间轴，无需漂移校正。
 */
export function useAudio() {
  let ctx: AudioContext | null = null
  let meta: { sampleRate: number; channels: number; sampleWidth: number } | null = null
  let nextTime = 0
  let started = false

  function init() {
    if (ctx) return
    ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 })
    nextTime = 0
    started = false
    meta = null
  }

  function setMeta(m: { sampleRate: number; channels: number; sampleWidth: number }) {
    meta = m
  }

  function playChunk(b64: string) {
    if (!ctx) return
    if (ctx.state === 'suspended') ctx.resume()

    // base64 → PCM16 bytes → Float32 samples
    const raw = atob(b64)
    const bytes = new Uint8Array(raw.length)
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i)
    const pcm16 = new Int16Array(bytes.buffer)
    const float32 = new Float32Array(pcm16.length)
    for (let i = 0; i < pcm16.length; i++) float32[i] = pcm16[i] / 32768.0

    const channels = meta?.channels || 1
    const sampleRate = meta?.sampleRate || 16000
    const framesPerCh = Math.floor(float32.length / channels)
    if (framesPerCh <= 0) return

    const buf = ctx.createBuffer(channels, framesPerCh, sampleRate)
    for (let ch = 0; ch < channels; ch++) {
      const data = buf.getChannelData(ch)
      for (let i = 0; i < framesPerCh; i++) {
        data[i] = float32[i * channels + ch]
      }
    }

    const src = ctx.createBufferSource()
    src.buffer = buf
    src.connect(ctx.destination)

    // 精确调度：上一段播完紧接着播下一段，无间隙无重叠
    const now = ctx.currentTime
    if (!started || nextTime < now) {
      nextTime = now + 0.05  // 首次留 50ms 启动余量
      started = true
    }
    src.start(nextTime)
    nextTime += buf.duration
  }

  function reset() {
    if (ctx) {
      try { ctx.close() } catch (_) {}
      ctx = null
    }
    nextTime = 0
    started = false
    meta = null
  }

  onUnmounted(reset)

  return { init, setMeta, playChunk, reset }
}
