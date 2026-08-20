<script setup lang="ts">
// VideoPanel: MSE fMP4 流式播放. front=主时钟(上报 currentTime); pop=从动(followTo 主时钟)

import { ref, watch, onBeforeUnmount, nextTick } from 'vue'

const props = withDefaults(defineProps<{
  title: string
  mediaUrl?: string
  isMuted?: boolean
  viewType: 'front' | 'pop'
  onProgressUpdate?: (sec: number) => void
  onEnded?: () => void
  isPlaying: boolean
  playbackRate?: number
  playbackSec?: number
}>(), {
  isMuted: false,
})

const videoRef = ref<HTMLVideoElement | null>(null)
let animationFrameId: number | null = null

// 逐帧循环: front 上报 currentTime(主时钟); pop 自驱动 followTo(主时钟)
function stepRenderLoop() {
  if (videoRef.value) {
    if (props.onProgressUpdate && !videoRef.value.paused) {
      props.onProgressUpdate(videoRef.value.currentTime)
    }
    if (props.viewType === 'pop' && props.playbackSec != null) {
      followTo(props.playbackSec)
    }
  }
  animationFrameId = requestAnimationFrame(stepRenderLoop)
}

function startRenderLoop() {
  if (animationFrameId === null) animationFrameId = requestAnimationFrame(stepRenderLoop)
}

function stopRenderLoop() {
  if (animationFrameId !== null) { cancelAnimationFrame(animationFrameId); animationFrameId = null }
}

function playVideo() {
  if (videoRef.value) {
    videoRef.value.currentTime = 0
    videoRef.value.play().catch(e => console.warn(`[${props.title}] 播放提示:`, e))
    startRenderLoop()
  }
}

function pauseVideo() {
  if (videoRef.value) videoRef.value.pause()
}

// pop 时刻对齐主时钟: 正常同速 diff≈0 不动; 失锁(>0.15s) buffered 内 seek 修正, 不变速追随
function followTo(masterSec: number) {
  const vid = videoRef.value
  if (!vid) return

  // stall 自恢复: pop paused 但应播放, 缓冲足够即唤醒
  if (vid.paused && props.isPlaying) {
    if (vid.buffered.length > 0) {
      const bufEnd = vid.buffered.end(vid.buffered.length - 1)
      if (bufEnd - vid.currentTime > 0.3) vid.play().catch(() => {})
    }
    return
  }

  if (Math.abs(vid.currentTime - masterSec) < 0.15) return
  // 偏差超 0.15s: masterSec 在 buffered 内则 seek 对齐, 否则不动(不上推末尾致卡死)
  for (let i = 0; i < vid.buffered.length; i++) {
    if (masterSec >= vid.buffered.start(i) && masterSec <= vid.buffered.end(i) - 0.1) {
      try { vid.currentTime = masterSec } catch {}
      return
    }
  }
}

// 标签页切回前台恢复播放
function handleVisibilityChange() {
  if (document.visibilityState === 'visible' && props.isPlaying && videoRef.value) {
    if (videoRef.value.paused) {
      videoRef.value.play().catch(() => {})
    }
  }
}

document.addEventListener('visibilitychange', handleVisibilityChange)

defineExpose({ pauseVideo })

// front/pop 同速共用 baseRate
watch(() => props.playbackRate, (rate) => {
  if (!videoRef.value || !rate || rate <= 0) return
  try { (videoRef.value as any).preservesPitch = true } catch {}
  videoRef.value.playbackRate = rate
}, { immediate: true })

watch(() => props.isPlaying, async (playing) => {
  await nextTick()
  if (!videoRef.value) return
  if (playing) playVideo()
  else pauseVideo()
})

startRenderLoop()

onBeforeUnmount(() => {
  stopRenderLoop()
  document.removeEventListener('visibilitychange', handleVisibilityChange)
})
</script>

<template>
  <div class="panel" style="flex:1">
    <div class="panel-title">{{ title }}</div>
    <div class="panel-body" style="padding:0;flex:1;position:relative">
      <video
        ref="videoRef"
        class="stream-video"
        :src="mediaUrl"
        :muted="isMuted"
        playsinline
        preload="auto"
        @ended="props.onEnded?.()"
      ></video>
    </div>
  </div>
</template>
