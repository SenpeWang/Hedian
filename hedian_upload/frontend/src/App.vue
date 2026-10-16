<script setup lang="ts">
// 装配层: 组合播放编排(usePlayback)与业务 store(useTranscript/useNotify/useReports)
// 消息流: WS → media 内核 → usePlayback 路由 → 业务 store(延迟注册回调)
// 模板消费经由 ws 聚合壳(保持模板绑定零改动)
import { ref, onMounted, onErrorCaptured } from 'vue'
import { usePlayback } from './composables/usePlayback'
import { useTranscript } from './composables/useTranscript'
import { useNotify } from './composables/useNotify'
import { useReports } from './composables/useReports'
import { resetPipeline } from './api/pipeline'
import { savePanelCache, loadPanelCache, clearPanelCache } from './media/panelCache'
import { createLogger } from './media/logger'
import HeaderBar from './components/HeaderBar.vue'
import VideoPanel from './components/VideoPanel.vue'
import VoicePanel from './components/VoicePanel.vue'
import NotifyPanel from './components/NotifyPanel.vue'
import ReportPanel from './components/ReportPanel.vue'

const log = createLogger('app')

// 主时钟(源视频秒): App 层创建, 播放编排与业务 store 共享
const currentPlaybackSec = ref(0)

const playback = usePlayback(currentPlaybackSec)
const transcript = useTranscript(playback.currentPlaybackSec)
const notify = useNotify(playback.currentPlaybackSec)
const reports = useReports()

// batch 事件解析(评估/flow/语音/人数/凝视; globalSec/sourceTimes 已由 usePlayback 消化)
function handleBatch(raw: Record<string, unknown>): void {
  const d = (raw.meta || raw) as Record<string, unknown>
  if (Array.isArray(d.flow_start)) for (const fe of d.flow_start) notify.addFlow(fe, true)
  if (Array.isArray(d.flow_end)) for (const fe of d.flow_end) notify.addFlow(fe, false)
  if (Array.isArray(d.flow_step)) for (const fs of d.flow_step) notify.addFlowStep(fs)
  if (Array.isArray(d.voice)) for (const v of d.voice) transcript.addVoice(v)
  if (Array.isArray(d.tracking)) for (const t of d.tracking) notify.addTracking(t)
  if (Array.isArray(d.gaze)) notify.addGaze(d.gaze)
  savePanels()
}

// 面板数据落本地缓存: 推理事件推一次即消费, 刷新后后端不补发, 全靠这里兜住
function savePanels(): void {
  savePanelCache({
    voice: transcript.rawEntries.value.map(v => ({ sec: v.sec, text: v.text, keys: v.keys || [] })),
    flow: notify.rawFlowEvents.value,
    cards: reports.finishedCards.value,
    counts: { sup: reports.supN.value, notice: reports.noticeN.value },
    scores: reports.segScores.value.slice(),
  })
}

// 内核 → 业务 store 路由注册(延迟注入破循环依赖)
playback.onBatch(handleBatch)
// 报告事件同样要落缓存(它不走 batch 通道, 走直推)
playback.onReport((msg) => { reports.handleReportEvent(msg); savePanels() })

// 刷新恢复: 把上轮已收到的对话/通知/评价从本地缓存回填(受播放进度过滤, 随播放逐步显示)
function restorePanels(): void {
  const cached = loadPanelCache()
  if (!cached) return
  transcript.restore(cached.voice || [])
  notify.restore(cached.flow || [])
  reports.restore(cached.cards || [], cached.counts, cached.scores || [])
  // 回看时刻: 刷新后 vis 缓存已被 /reset 清掉, 视频源失效、播放时钟停在 0,
  // 而三个面板都是按「sec ≤ 当前播放时刻」过滤的, 时钟为 0 会把刚恢复的数据
  // 全部挡掉(表现依旧一片空白)。这里把时钟推到缓存数据的最后时刻, 让历史内容
  // 直接可见; 新一轮开始时 resetAll 会把它归零。
  const lastSec = Math.max(
    0,
    ...(cached.voice || []).map(v => Number(v.sec) || 0),
    ...(cached.flow || []).map(f => Number(f.sec) || 0),
    ...(cached.cards || []).map(c => Number(c.endSec) || Number(c.startSec) || 0),
  )
  if (lastSec > 0) currentPlaybackSec.value = lastSec
  log.info(
    `已恢复上轮面板数据: 对话 ${cached.voice?.length || 0} 条, 通知 ${cached.flow?.length || 0} 条, 评价 ${cached.cards?.length || 0} 份`
  )
}

// 全量重置: 播放编排 + 三个业务 store
function resetAll(): void {
  playback.resetState()
  transcript.reset()
  notify.reset()
  reports.reset()
}

const frontPanelRef = ref<InstanceType<typeof VideoPanel> | null>(null)
const popPanelRef = ref<InstanceType<typeof VideoPanel> | null>(null)

// 页面加载/刷新即重置:kill 推理子进程 + 清空状态,保证每次从干净状态开始
onMounted(() => {
  resetPipeline().catch(() => { /* 失败也继续本地重置 */ }).finally(() => {
    resetAll()
    // 刷新后从本地缓存回填上轮面板数据(新一轮开始时由 handleStart 清空)
    restorePanels()
  })
})

function handleStart() {
  clearPanelCache()
  // resetState 会重建 MSE 并设新 mediaUrl(触发 video new load);
  // play 交给 isPlaying watch 在 nextTick 后触发,确保 src 已变,
  // 避免 play() 被 new load 中断(AbortError)
  resetAll()
  playback.startPipeline()
}

function handleStop() {
  frontPanelRef.value?.pauseVideo()
  popPanelRef.value?.pauseVideo()
  playback.stopPipeline()
}

// front 主时钟: emit progress 设 currentPlaybackSec(pop 自驱动跟随, 不在此调)
function handleFrontProgress(sec: number) {
  playback.reportPlaybackProgress(sec)
}
// front 结束暂停 pop(主从锁步, 不依赖硬编码时长)
function handleFrontEnded() {
  popPanelRef.value?.pauseVideo()
}

// 错误边界: 捕获子组件异常, 落日志不崩 UI(不新增 UI 元素, 守 CSS 红线)
onErrorCaptured((err, _instance, info) => {
  log.error(`组件异常(${info})`, err)
  return false
})

// 模板消费聚合壳(与旧 useWS 返回形状一致, 模板绑定零改动)
const ws = {
  status: playback.status,
  progress: playback.progress,
  isPlaying: playback.isPlaying,
  fmt: playback.fmt,
  frontMediaUrl: playback.frontMediaUrl,
  popMediaUrl: playback.popMediaUrl,
  playbackRate: playback.playbackRate,
  currentPlaybackSec: playback.currentPlaybackSec,
  voiceEntries: transcript.voiceEntries,
  people: notify.people,
  gaze: notify.gaze,
  flowEvents: notify.flowEvents,
  segCards: reports.orderedCards,
  supN: reports.supN,
  noticeN: reports.noticeN,
  totalScore: reports.totalScore,
  avgScore: reports.avgScore,
  toggleCard: reports.toggleCard,
  resetState: resetAll,
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
        :is-playing="ws.isPlaying.value"
        :playback-rate="ws.playbackRate.value"
        @progress="handleFrontProgress"
        @ended="handleFrontEnded"
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
      :notice-n="ws.noticeN.value"
      :total="ws.totalScore.value"
      :avg="ws.avgScore.value"
      @toggle="ws.toggleCard"
    />
  </div>
</template>
