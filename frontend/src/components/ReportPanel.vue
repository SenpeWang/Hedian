<script setup lang="ts">
import { nextTick, watch, ref, onBeforeUnmount } from 'vue'
import type { SegCard } from '../types'

const props = defineProps<{
  segCards: SegCard[]
  supN: number
  ticketN: number
  noticeN: number
  total: number
  avg: string
}>()

const cardsEl = ref<HTMLElement | null>(null)

async function scrollToBottom() {
  await nextTick()
  if (cardsEl.value) cardsEl.value.scrollTop = cardsEl.value.scrollHeight
}

// 卡片新增: 初始化打字机长度 + 滚底
watch(() => props.segCards.length, () => { scrollToBottom() })

// 流式内容变化: 滚底(打字机由 rAF 驱动)
watch(() => props.segCards.map(c => (c.streamBuffer || '') + (c.reportText || '')).join('\n'), () => {
  scrollToBottom()
})

function parseReportContent(text: string) {
  if (!text) return { think: '', report: '' }
  const thinkMatch = text.match(/<think>([\s\S]*?)(?:<\/think>|$)/i)
  let think = thinkMatch ? thinkMatch[1].trim() : ''
  let report = text.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '').trim()
  return { think, report }
}

// 前端模拟流式逐字: streamBuffer 按 chunk 累积(后端到达), shownLen RAF 逐字追赶, 一字一字显示
const shownLen = ref<Record<string, number>>({})

function shownText(card: SegCard, field: 'think' | 'report'): string {
  const raw = card.streamBuffer || card.reportText || ''
  if (!raw) return ''
  if (!card.streaming) return parseReportContent(raw)[field]  // 完成显示完整
  // 先按 shownLen 截 streamBuffer(整体进度), 再 parse 出对应 field — think+report 都逐字
  const len = shownLen.value[card.flowId] || 0
  return parseReportContent(raw.slice(0, Math.min(len, raw.length)))[field]
}

// 慢速逐字打字机: setInterval 60ms/字(~16字/秒), 模拟大模型流式输出
// 前端收到完整数据, 逐字慢速展示
let timerId: ReturnType<typeof setInterval> | null = null
function typewriterTick() {
  let needContinue = false
  for (const card of props.segCards) {
    if (!card.streaming) continue
    const full = card.streamBuffer || card.reportText || ''
    const cur = shownLen.value[card.flowId] || 0
    if (cur < full.length) {
      shownLen.value[card.flowId] = Math.min(cur + 1, full.length)
      needContinue = true
    } else if (card.reportText) {
      // 已逐字追赶到末尾 + 最终报告(segment_report)已到达 → 切完成态(显分数/进度条/完成图标)
      card.streaming = false
    }
  }
  scrollToBottom()  // 逐字滚底跟随最新输出(同 VoicePanel watch 实时 text)
  if (!needContinue && timerId !== null) { clearInterval(timerId); timerId = null }
}

// streaming 卡片存在 → 启动打字机
watch(() => props.segCards.some(c => c.streaming), (has) => {
  if (has && timerId === null) timerId = setInterval(typewriterTick, 60)
}, { immediate: true })
// segment_report 到达(reportText 变化)时若打字机已停但仍有 streaming 卡片 → 重启跑一次完成态切换
// (streamBuffer 早追完已停, segment_report 后到, 需 tick 触发 streaming=false)
watch(() => props.segCards.map(c => c.reportText).join('|'), () => {
  if (timerId === null && props.segCards.some(c => c.streaming)) timerId = setInterval(typewriterTick, 60)
})

onBeforeUnmount(() => { if (timerId !== null) clearInterval(timerId) })

function scoreColor(score: number) {
  return score >= 8 ? '#00ff88' : score >= 5 ? '#ffaa00' : '#ff4d4d'
}

function borderColor(flowType: string) {
  if (flowType === 'supervision') return '#00d4ff'
  if (flowType === 'info_notice') return '#00ffcc'
  return '#ffaa00'
}

function cardIcon(flowType: string) {
  if (flowType === 'supervision') return '🛡️'
  if (flowType === 'info_notice') return '📢'
  return '🎫'
}

function cardLabel(flowType: string) {
  if (flowType === 'supervision') return '监护制'
  if (flowType === 'info_notice') return '信息通报'
  return '自唱票'
}

function toggle(card: SegCard) {
  card.collapsed = !card.collapsed
}
</script>

<template>
  <div class="panel">
    <div class="panel-title">📋 流程评价</div>
    <div class="panel-body" ref="cardsEl" style="scroll-behavior: smooth">
      <div class="summary-grid">
        <div><div class="val">{{ supN }}</div><div class="lbl">监护制流程</div></div>
        <div><div class="val">{{ ticketN }}</div><div class="lbl">自唱票流程</div></div>
        <div><div class="val">{{ noticeN }}</div><div class="lbl">信息通报</div></div>
        <div><div class="val">{{ avg }}</div><div class="lbl">平均分</div></div>
        <div><div class="val">{{ total }}</div><div class="lbl">总分</div></div>
      </div>
      <div>
        <div v-for="card in segCards" :key="card.flowId"
             class="seg-card" :class="{ collapsed: card.collapsed }"
             :style="{ borderLeftColor: card.streaming ? '#6b7a90' : borderColor(card.flowType) }">
          <div class="sc-head" @click="toggle(card)">
            <span>
              {{ card.streaming ? '🤖' : cardIcon(card.flowType) }}
              {{ cardLabel(card.flowType) }} #{{ card.flowId }}{{ card.streaming ? '' : ' [' + card.continueSec + 's]' }}
              <span class="sc-toggle-icon">{{ card.collapsed ? '▶' : '▼' }}</span>
            </span>
            <span v-if="!card.streaming" class="sc-score" :style="{ color: scoreColor(card.score) }">{{ card.score }}/10</span>
          </div>

          <div class="sc-bar" v-if="!card.streaming">
            <div class="sc-bar-fill" :style="{ width: card.score * 10 + '%', background: scoreColor(card.score) }"></div>
          </div>

          <!-- 🧠 大模型思考推理过程展示框 (打字中与打字完成全时段常驻显示) -->
          <div v-if="parseReportContent(card.streamBuffer || card.reportText).think" class="think-block">
            <div class="think-title">🧠 思考推理过程</div>
            <div class="think-body">{{ shownText(card, 'think') }}</div>
          </div>

          <!-- 📋 正式评价报告正文 -->
          <div class="sc-detail">{{ shownText(card, 'report') }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.think-block {
  background: rgba(0, 212, 255, 0.06);
  border: 1px solid rgba(0, 212, 255, 0.2);
  border-radius: 4px;
  padding: 5px 8px;
  margin: 4px 0 6px 0;
}
.think-title {
  font-size: 10px;
  font-weight: 700;
  color: #00d4ff;
  margin-bottom: 2px;
}
.think-body {
  font-size: 9px;
  color: #a0b0c0;
  line-height: 1.35;
  white-space: pre-wrap;
}

</style>
