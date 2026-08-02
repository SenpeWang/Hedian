<script setup lang="ts">
import { useWS } from './composables/useWS'
import HeaderBar from './components/HeaderBar.vue'
import VideoPanel from './components/VideoPanel.vue'
import VoicePanel from './components/VoicePanel.vue'
import NotifyPanel from './components/NotifyPanel.vue'
import ReportPanel from './components/ReportPanel.vue'

const ws = useWS()

function handleStart() {
  ws.startPipeline()
}
</script>

<template>
  <audio ref="audioRef" id="mainAudio" preload="auto" style="display:none"></audio>
  <HeaderBar
    :status="ws.status.value"
    :status-text="ws.statusText.value"
    :progress="ws.progress"
    @start="handleStart"
  />

  <div class="main">
    <div class="video-col tracker">
      <VideoPanel title="🎥 目标跟踪 (front)" :frame-src="ws.frameFront.value" :has-frame="ws.hasFrameFront.value" />
    </div>
    <div class="video-col">
      <VideoPanel title="🔍 屏幕检测 (bup)" :frame-src="ws.frameBup.value" :has-frame="ws.hasFrameBup.value" />
    </div>
    <div class="video-col">
      <VideoPanel title="🔍 文件检测 (pop)" :frame-src="ws.framePop.value" :has-frame="ws.hasFramePop.value" />
    </div>
  </div>

  <div class="bottom">
    <VoicePanel :entries="ws.voiceEntries.value" :fmt="ws.fmt" />
    <NotifyPanel :people="ws.people" :gaze="ws.gaze" :flow-events="ws.flowEvents.value" :fmt="ws.fmt" />
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
