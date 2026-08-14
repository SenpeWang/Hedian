<script setup lang="ts">
import { nextTick, watch, ref } from 'vue'
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
const thinkCollapsedMap = ref<Record<string, boolean>>({})

// 新卡片增加或流式文字更新时，自动平滑滚动到底部活跃卡片
watch(
  () => [props.segCards.length, props.segCards.map(c => c.streamBuffer).join('')],
  async () => {
    await nextTick()
    if (cardsEl.value) {
      cardsEl.value.scrollTo({
        top: cardsEl.value.scrollHeight,
        behavior: 'smooth',
      })
    }
  },
  { deep: true, immediate: true }
)

function parseReportContent(text: string) {
  if (!text) return { think: '', report: '' }
  const thinkMatch = text.match(/<think>([\s\S]*?)(?:<\/think>|$)/i)
  let think = thinkMatch ? thinkMatch[1].trim() : ''
  let report = text.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '').trim()
  return { think, report }
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
  if (flowType === 'supervision') return '监护制流程'
  if (flowType === 'info_notice') return '信息通报流程'
  return '自唱票流程'
}

function toggleCard(card: SegCard) {
  card.collapsed = !card.collapsed
}

function toggleThink(flowId: string, event: Event) {
  event.stopPropagation()
  thinkCollapsedMap.value[flowId] = !thinkCollapsedMap.value[flowId]
}
</script>

<template>
  <div class="panel">
    <div class="panel-title">
      <span>📋 流程合规评价</span>
    </div>
    <div class="panel-body" ref="cardsEl">
      <!-- 汇总数据指标网格 -->
      <div class="summary-grid">
        <div class="summary-cell"><div class="val text-cyan">{{ supN }}</div><div class="lbl">监护制</div></div>
        <div class="summary-cell"><div class="val text-orange">{{ ticketN }}</div><div class="lbl">自唱票</div></div>
        <div class="summary-cell"><div class="val text-green">{{ noticeN }}</div><div class="lbl">信息通报</div></div>
        <div class="summary-cell"><div class="val text-blue">{{ avg }}</div><div class="lbl">平均分</div></div>
        <div class="summary-cell"><div class="val text-purple">{{ total }}</div><div class="lbl">总分</div></div>
      </div>

      <!-- 评估卡片列表 -->
      <div class="cards-wrapper">
        <div
          v-for="card in segCards"
          :key="card.flowId"
          class="seg-card"
          :class="{
            collapsed: card.collapsed,
            'streaming-card': card.streaming,
          }"
          :style="{ borderLeftColor: card.streaming ? '#00d4ff' : borderColor(card.flowType) }"
        >
          <!-- 卡片头部：简洁清晰，无多余的分数瑕疵徽章 -->
          <div class="sc-head" @click="toggleCard(card)">
            <div class="sc-title-group">
              <span class="sc-icon">{{ card.streaming ? '⚡' : cardIcon(card.flowType) }}</span>
              <span class="sc-name">
                {{ card.streaming ? '大模型流式推理中…' : cardLabel(card.flowType) + ' #' + card.flowId }}
              </span>
              <span class="sc-duration">[{{ card.continueSec }}s]</span>
              <span class="sc-toggle-icon">{{ card.collapsed ? '▶' : '▼' }}</span>
            </div>
            <div v-if="card.streaming" class="sc-streaming-tag">
              <span class="pulse-dot"></span> 生成中
            </div>
          </div>

          <!-- 🧠 大模型深度思考推理过程 -->
          <div
            v-if="parseReportContent(card.streamBuffer || card.reportText).think"
            class="think-block"
            :class="{ 'think-collapsed': thinkCollapsedMap[card.flowId] }"
          >
            <div class="think-header" @click="toggleThink(card.flowId, $event)">
              <div class="think-title">
                <span class="think-icon">🧠</span>
                <span>模型思考链推理 (Reasoning Log)</span>
                <span v-if="card.streaming" class="think-pulse-tag">思考中...</span>
              </div>
              <span class="think-toggle-btn">{{ thinkCollapsedMap[card.flowId] ? '展开 ▶' : '收起 ▼' }}</span>
            </div>
            <div v-show="!thinkCollapsedMap[card.flowId]" class="think-body">
              {{ parseReportContent(card.streamBuffer || card.reportText).think }}
              <span v-if="card.streaming" class="cursor-blink">▊</span>
            </div>
          </div>

          <!-- 📋 最终正式评价报告正文 -->
          <div
            v-if="parseReportContent(card.streamBuffer || card.reportText).report || card.streaming"
            class="sc-detail"
          >
            <div class="detail-label">📋 评估结论：</div>
            <div class="detail-content">
              {{ parseReportContent(card.streamBuffer || card.reportText).report }}
              <span v-if="card.streaming && !parseReportContent(card.streamBuffer || card.reportText).think" class="cursor-blink">▊</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.summary-cell {
  background: rgba(10, 14, 26, 0.8) !important;
  border: 1px solid #1e2a42;
  border-radius: 4px;
}
.text-cyan { color: #00d4ff; }
.text-orange { color: #ffaa00; }
.text-green { color: #00ff88; }
.text-blue { color: #3b82f6; }
.text-purple { color: #a855f7; }

.cards-wrapper {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.seg-card {
  background: rgba(10, 14, 26, 0.85);
  border: 1px solid #1e2a42;
  border-left-width: 4px;
  border-radius: 6px;
  padding: 8px 10px;
  transition: all 0.3s ease;
}
.streaming-card {
  box-shadow: 0 0 12px rgba(0, 212, 255, 0.25);
  border-color: rgba(0, 212, 255, 0.5);
}
.sc-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
  cursor: pointer;
  user-select: none;
}
.sc-title-group {
  display: flex;
  align-items: center;
  gap: 6px;
}
.sc-icon {
  font-size: 15px;
}
.sc-name {
  font-weight: 600;
  font-size: 14px;
  color: #e0e6f0;
}
.sc-duration {
  font-size: 12px;
  color: #8899aa;
}
.sc-streaming-tag {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: #00d4ff;
}
.pulse-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #00d4ff;
  animation: pulse 1s infinite;
}

.think-block {
  background: rgba(0, 212, 255, 0.04);
  border: 1px solid rgba(0, 212, 255, 0.2);
  border-radius: 5px;
  margin: 6px 0;
  overflow: hidden;
}
.think-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 5px 8px;
  background: rgba(0, 212, 255, 0.08);
  cursor: pointer;
  user-select: none;
}
.think-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
  color: #00d4ff;
}
.think-pulse-tag {
  font-size: 11px;
  color: #ffaa00;
  margin-left: 6px;
}
.think-toggle-btn {
  font-size: 12px;
  color: #8899aa;
}
.think-body {
  padding: 8px 10px;
  font-size: 13px;
  color: #cbd5e1;
  line-height: 1.6;
  white-space: pre-wrap;
  font-family: 'JetBrains Mono', 'Consolas', monospace;
  background: rgba(0, 0, 0, 0.25);
  border-top: 1px solid rgba(0, 212, 255, 0.12);
}

.sc-detail {
  margin-top: 6px;
  font-size: 14px;
  line-height: 1.6;
}
.detail-label {
  font-weight: 700;
  font-size: 14px;
  color: #00ffcc;
  margin-bottom: 3px;
}
.detail-content {
  color: #e2e8f0;
  white-space: pre-wrap;
  padding: 4px 6px;
  font-size: 14px;
  line-height: 1.6;
}

.cursor-blink {
  display: inline-block;
  color: #00d4ff;
  animation: blink 0.8s infinite;
  font-weight: bold;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style>
