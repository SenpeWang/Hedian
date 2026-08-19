<script setup lang="ts">
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
}>(), {
  isMuted: false,
})

const videoRef = ref<HTMLVideoElement | null>(null)
let animationFrameId: number | null = null

// 渲染循环:每帧上报当前播放时间(驱动主时钟 → 结构化面板按时刻取数 + 双流锁步)
// 标注已画进帧(后端 fMP4),前端不再做 Canvas 叠加绘制
function stepRenderLoop() {
  if (videoRef.value && props.onProgressUpdate && !videoRef.value.paused) {
    props.onProgressUpdate(videoRef.value.currentTime)
  }
  animationFrameId = requestAnimationFrame(stepRenderLoop)
}

function startRenderLoop() {
  if (animationFrameId === null) {
    animationFrameId = requestAnimationFrame(stepRenderLoop)
  }
}

function stopRenderLoop() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId)
    animationFrameId = null
  }
}

function playVideo() {
  if (videoRef.value) {
    videoRef.value.currentTime = 0
    videoRef.value.play().catch(e => console.warn(`[${props.title}] 播放提示:`, e))
    startRenderLoop()
  }
}

function pauseVideo() {
  if (videoRef.value) {
    videoRef.value.pause()
  }
}

/** 主从追随:根据与主时钟(front)的偏差, 平滑调 playbackRate 追随, 不离散seek.
 *  偏差小:同速; 落后:略加速; 超前:略减速; 严重失步:兜底seek.
 *  baseRate=props.playbackRate(最慢板块倍率), 追随在其基础上±微调. */
function followTo(sec: number) {
  if (!videoRef.value) return
  const diff = videoRef.value.currentTime - sec  // 正=pop超前, 负=pop落后
  const base = props.playbackRate || 1.0
  // 检查 front 时刻是否在 pop 已缓冲范围内(避免 seek 到未缓冲位置 → 乱跳帧)
  const buf = videoRef.value.buffered
  let inBuffer = false
  for (let i = 0; i < buf.length; i++) {
    if (sec >= buf.start(i) - 0.05 && sec <= buf.end(i) - 0.3) { inBuffer = true; break }
  }
  if (!inBuffer) {
    // front 时刻未缓冲(pop 段未到): 暂停等, 不 seek 不乱跳
    videoRef.value.playbackRate = 0
    return
  }
  // 严格对齐: 偏差>0.1 即 seek 到 front.currentTime(buffered 内安全), 不用 rate 追随避免累积偏差
  if (Math.abs(diff) > 0.1) {
    try { videoRef.value.currentTime = sec } catch {}
  }
  videoRef.value.playbackRate = base  // 同速率播放(与 front 同速)
}

defineExpose({ playVideo, pauseVideo, followTo, currentTime: () => videoRef.value?.currentTime ?? 0, duration: () => videoRef.value?.duration ?? 0 })

watch(() => props.playbackRate, (rate) => {
  if (videoRef.value && rate && rate > 0) {
    try { (videoRef.value as any).preservesPitch = true } catch {}
    videoRef.value.playbackRate = rate
  }
}, { immediate: true })

watch(() => props.isPlaying, async (playing) => {
  await nextTick()
  if (!videoRef.value) return
  if (playing) {
    playVideo()
  } else {
    pauseVideo()
  }
})

startRenderLoop()

onBeforeUnmount(() => {
  stopRenderLoop()
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
