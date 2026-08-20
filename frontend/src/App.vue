<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useWS } from './composables/useWS'
import HeaderBar from './components/HeaderBar.vue'
import VideoPanel from './components/VideoPanel.vue'
import VoicePanel from './components/VoicePanel.vue'
import NotifyPanel from './components/NotifyPanel.vue'
import ReportPanel from './components/ReportPanel.vue'

const ws = useWS()

const frontPanelRef = ref<InstanceType<typeof VideoPanel> | null>(null)
const popPanelRef = ref<InstanceType<typeof VideoPanel> | null>(null)

// 页面加载/刷新即重置:kill 推理子进程 + 清空状态,保证每次从干净状态开始
onMounted(() => {
  fetch('/reset', { method: 'POST' }).then(() => { ws.resetState() }).catch(() => { ws.resetState() })
  // 注入各视角 video.currentTime,供 MSE SourceBuffer trim 清理已播放数据(防 QuotaExceededError)
  ws.setClockFns(
    () => frontPanelRef.value?.currentTime() ?? 0,
    () => popPanelRef.value?.currentTime() ?? 0,
  )
})

function handleStart() {
  // startPipeline 的 resetState 会重建 MSE 并设新 mediaUrl(触发 video new load);
  // play 交给 isPlaying watch 在 nextTick 后触发,确保 src 已变,
  // 避免 play() 被 new load 中断(AbortError)
  ws.startPipeline()
}

function handleStop() {
  frontPanelRef.value?.pauseVideo()
  popPanelRef.value?.pauseVideo()
  ws.stopPipeline()
}

// front 主时钟: 上报设 currentPlaybackSec(pop 自驱动跟随, 不在此调)
function handleFrontProgress(sec: number) {
  ws.reportPlaybackProgress(sec)
}
// front 结束暂停 pop(641<644 防错位)
function handleFrontEnded() {
  popPanelRef.value?.pauseVideo()
}
</script>

<template>
  <HeaderBar
    :status="ws.status.value"
    :progress="ws.progress.value"
    @start="handleStart"
    @stop="handleStop"
  />

  <div class="main">
    <div class="video-col">
      <VideoPanel
        ref="frontPanelRef"
        title="camFRONT"
        :media-url="ws.frontMediaUrl.value"
        :is-muted="false"
        view-type="front"
        :on-progress-update="handleFrontProgress"
        :on-ended="handleFrontEnded"
        :is-playing="ws.isPlaying.value"
        :playback-rate="ws.playbackRate.value"
      />
    </div>
    <div class="video-col">
      <VideoPanel
        ref="popPanelRef"
        title="camPOP"
        :media-url="ws.popMediaUrl.value"
        :is-muted="true"
        view-type="pop"
        :is-playing="ws.isPlaying.value"
        :playback-rate="ws.playbackRate.value"
        :playback-sec="ws.currentPlaybackSec.value"
      />
    </div>
  </div>

  <div class="bottom">
    <VoicePanel :entries="ws.voiceEntries.value" :fmt="ws.fmt" />
    <NotifyPanel
      :people="ws.people.value"
      :gaze="ws.gaze.value"
      :flow-events="ws.flowEvents.value"
      :fmt="ws.fmt"
    />
    <ReportPanel
      :seg-cards="ws.segCards.value"
      :sup-n="ws.supN.value"
      :ticket-n="ws.ticketN.value"
      :notice-n="ws.noticeN.value"
      :total="ws.totalCount()"
      :avg="ws.avgScore()"
    />
  </div>
</template>
