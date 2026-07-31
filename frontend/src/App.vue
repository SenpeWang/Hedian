<script setup lang="ts">
import { useSSE } from './composables/useSSE'
import HeaderBar from './components/HeaderBar.vue'
import VideoPanel from './components/VideoPanel.vue'
import VoicePanel from './components/VoicePanel.vue'
import NotifyPanel from './components/NotifyPanel.vue'
import ReportPanel from './components/ReportPanel.vue'

const sse = useSSE()

function handleStart() {
  sse.startPipeline()
}
</script>

<template>
  <audio ref="audioRef" id="mainAudio" preload="auto" style="display:none"></audio>
  <HeaderBar
    :status="sse.status.value"
    :status-text="sse.statusText.value"
    :progress="sse.progress"
    @start="handleStart"
  />

  <div class="main">
    <div class="video-col tracker">
      <VideoPanel title="🎥 目标跟踪 (front)" :frame-src="sse.frameFront.value" :has-frame="sse.hasFrameFront.value" />
    </div>
    <div class="video-col">
      <VideoPanel title="🔍 屏幕检测 (bup)" :frame-src="sse.frameBup.value" :has-frame="sse.hasFrameBup.value" />
    </div>
    <div class="video-col">
      <VideoPanel title="🔍 文件检测 (pop)" :frame-src="sse.framePop.value" :has-frame="sse.hasFramePop.value" />
    </div>
  </div>

  <div class="bottom">
    <VoicePanel :entries="sse.voiceEntries.value" :fmt="sse.fmt" />
    <NotifyPanel :people="sse.people" :gaze="sse.gaze" :flow-events="sse.flowEvents.value" :fmt="sse.fmt" />
    <ReportPanel
      :seg-cards="sse.segCards.value"
      :sup-n="sse.supN.value"
      :ticket-n="sse.ticketN.value"
      :notice-n="sse.noticeN.value"
      :total="sse.totalCount()"
      :avg="sse.avgScore()"
    />
  </div>
</template>
