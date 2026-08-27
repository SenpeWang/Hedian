<script setup lang="ts">
// VideoPanel: MSE fMP4 流式播放展示组件(薄)
// front=主时钟(emit currentTime); pop=从动(锁步算法在 media/slave-sync, 此处仅驱动)
import { ref, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { createSlaveSyncState, syncSlave, resetSlaveSync } from '../media/slave-sync'

const props = withDefaults(defineProps<{
  title: string
  mediaUrl?: string
  isMuted?: boolean
  viewType: 'front' | 'pop'
  isPlaying: boolean
  playbackRate?: number
  playbackSec?: number
}>(), {
  isMuted: false,
})

const emit = defineEmits<{
  progress: [sec: number]
  ended: []
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
let animationFrameId: number | null = null
const slaveState = createSlaveSyncState()
// 起播门槛: 未起播过需攒 2s jitter buffer 再播(防首段即播→断供→播停循环, 声音卡顿);
// 已起播后 stall 恢复只需 0.3s 余量(快速恢复)
let startedOnce = false

// 逐帧循环: front emit currentTime(主时钟); pop 调 slave-sync 锁步. 仅 isPlaying 时跑
function stepRenderLoop() {
  const vid = videoRef.value
  if (vid) {
    if (props.viewType === 'front') {
      // stall 自恢复: 首次 play() 时 MSE 数据常未就绪会 reject, buffered 足够后 RAF 持续重试 play
      if (vid.paused && props.isPlaying && vid.buffered.length > 0) {
        const bufEnd = vid.buffered.end(vid.buffered.length - 1)
        const need = startedOnce ? 0.3 : 2
        if (bufEnd - vid.currentTime > need) {
          vid.play().then(() => { startedOnce = true }).catch(() => {})
        }
      }
      if (!vid.paused) emit('progress', vid.currentTime)
    } else if (props.viewType === 'pop' && props.playbackSec != null) {
      syncSlave(vid, props.playbackSec, props.isPlaying, props.playbackRate ?? 1, slaveState)
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

// play() + 启动 RAF; 不重置 currentTime(isPlaying 抖动/恢复时不打回 0, 重建由 watch mediaUrl)
function playVideo() {
  if (videoRef.value) {
    videoRef.value.play().catch(e => console.warn(`[${props.title}] 播放提示:`, e))
    startRenderLoop()
  }
}

function pauseVideo() {
  if (videoRef.value) videoRef.value.pause()
}

// 标签页切回前台恢复播放
function handleVisibilityChange() {
  if (document.visibilityState === 'visible' && props.isPlaying && videoRef.value) {
    if (videoRef.value.paused) videoRef.value.play().catch(() => {})
  }
}

defineExpose({ pauseVideo })

// 速率控制权分离: front 速率归水位引擎(此处 watch 设置);
// pop 速率完全归 slave-sync 伺服层(每帧 base×factor 连续修正, 此处不得干预——
// 否则伺服修正被 watch 周期性清零, pop 追赶失效掉队累积)
watch(() => props.playbackRate, (rate) => {
  if (props.viewType !== 'front') return
  if (!videoRef.value || !rate || rate <= 0) return
  videoRef.value.preservesPitch = true
  videoRef.value.playbackRate = rate
})

// mediaUrl 变化(重建流)时重置 currentTime=0 + 加载 + 清锁步越界计时 + 重置起播门槛
watch(() => props.mediaUrl, () => {
  resetSlaveSync(slaveState)
  startedOnce = false
  if (videoRef.value) videoRef.value.currentTime = 0
})

// isPlaying 门控: playing 启动 RAF+播放, 停止即停 RAF(空闲不空转)
watch(() => props.isPlaying, async (playing) => {
  await nextTick()
  if (!videoRef.value) return
  if (playing) playVideo()
  else { pauseVideo(); stopRenderLoop() }
}, { immediate: true })

onMounted(() => {
  document.addEventListener('visibilitychange', handleVisibilityChange)
})

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
        @ended="emit('ended')"
      ></video>
    </div>
  </div>
</template>
