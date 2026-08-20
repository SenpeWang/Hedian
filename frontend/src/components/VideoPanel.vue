<script setup lang="ts">
/**
 * @fileoverview 视频展示面板组件 (VideoPanel).
 *
 * 实现了基于 MSE (MediaSource Extensions) 的 fMP4 视频流实时解码播放，
 * 并针对多视角对齐场景提供了基于缓冲水位门控的时刻对齐主从同步控制。
 */

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

/**
 * 驱动前端主时钟的逐帧渲染循环.
 *
 * 通过 requestAnimationFrame 获取当前视频硬件解码时间戳，
 * 向上层容器实时上报以驱动所有结构化面板与双流锁步。
 */
function stepRenderLoop() {
  if (videoRef.value) {
    if (props.onProgressUpdate && !videoRef.value.paused) {
      props.onProgressUpdate(videoRef.value.currentTime)
    }
    // pop 自驱动跟随主时钟: front underrun paused 时 currentPlaybackSec 冻结,
    // pop 仍每帧 followTo(主时钟) seek 回, 不超前不跑末尾(不依赖 handleFrontProgress 上报)
    if (props.viewType === 'pop' && props.playbackSec != null) {
      followTo(props.playbackSec)
    }
  }
  animationFrameId = requestAnimationFrame(stepRenderLoop)
}

/**
 * 启动逐帧渲染循环.
 */
function startRenderLoop() {
  if (animationFrameId === null) {
    animationFrameId = requestAnimationFrame(stepRenderLoop)
  }
}

/**
 * 停止逐帧渲染循环.
 */
function stopRenderLoop() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId)
    animationFrameId = null
  }
}

/**
 * 重置并开始播放当前视频.
 */
function playVideo() {
  if (videoRef.value) {
    videoRef.value.currentTime = 0
    videoRef.value.play().catch(e => console.warn(`[${props.title}] 播放提示:`, e))
    startRenderLoop()
  }
}

/**
 * 暂停当前视频播放.
 */
function pauseVideo() {
  if (videoRef.value) {
    videoRef.value.pause()
  }
}

/**
 * 主从时刻对齐 (buffered-seek time alignment).
 *
 * 工业多路 MSE 对齐: pop 与 front 同速 (共用 baseRate), 偏差只靠 buffered 内
 * seek 对齐时刻修正, 不变速追随 (变速在离散 append 的 MSE buffer 上会累积漂移).
 * 严重失步兜底 seek 优先落到 buffered 内, 否则回自己 buffer 末尾配合 front 减速收敛.
 *
 * Args:
 *   masterSec: 主视角 (front) 当前视频播放时刻 (秒).
 */
function followTo(masterSec: number) {
  const vid = videoRef.value
  if (!vid) return

  // stall 自恢复: pop 暂停但应播放, 缓冲恢复足够即唤醒
  if (vid.paused && props.isPlaying) {
    if (vid.buffered.length > 0) {
      const bufEnd = vid.buffered.end(vid.buffered.length - 1)
      if (bufEnd - vid.currentTime > 0.3) {
        vid.play().catch(() => {})
      }
    }
    return
  }

  const diff = vid.currentTime - masterSec // 正: pop 超前, 负: pop 落后
  const absDiff = Math.abs(diff)
  if (absDiff < 0.15) return // 自然跟随, 不动

  // 时刻是否落在 pop 已缓冲区间内 (避免 seek 到未缓冲区乱跳帧)
  const inBuf = (t: number) => {
    for (let i = 0; i < vid.buffered.length; i++) {
      if (t >= vid.buffered.start(i) && t <= vid.buffered.end(i) - 0.1) return true
    }
    return false
  }

  if (absDiff < 1.0) {
    // 中等偏差: buffered 内 seek 对齐, 否则不动等缓冲
    if (inBuf(masterSec)) { try { vid.currentTime = masterSec } catch {} }
    return
  }

  // 严重失步兜底: masterSec 在 buffered 内则 seek 对齐, 否则保持不动(绝不上推末尾致卡死)
  if (inBuf(masterSec)) { try { vid.currentTime = masterSec } catch {} }
}

/**
 * 标签页切前台自适应恢复处理.
 */
function handleVisibilityChange() {
  if (document.visibilityState === 'visible' && props.isPlaying && videoRef.value) {
    if (videoRef.value.paused) {
      videoRef.value.play().catch(() => {})
    }
  }
}

document.addEventListener('visibilitychange', handleVisibilityChange)

defineExpose({
  playVideo,
  pauseVideo,
  followTo,
  currentTime: () => videoRef.value?.currentTime ?? 0,
  duration: () => videoRef.value?.duration ?? 0,
})

// 监听全局播放倍率: front/pop 同速 (共用 baseRate), 不再有 pop 自行调速
watch(() => props.playbackRate, (rate) => {
  if (!videoRef.value || !rate || rate <= 0) return
  try { (videoRef.value as any).preservesPitch = true } catch {}
  videoRef.value.playbackRate = rate
}, { immediate: true })

// 监听播放状态切换
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
