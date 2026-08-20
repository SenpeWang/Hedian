<script setup lang="ts">
import { ref, toRef } from 'vue'
import type { SegCard } from '../types'
import { useScrollBottom } from '../composables/useScrollBottom'
import { useTypewriter } from '../composables/useTypewriter'

const props = defineProps<{
  segCards: SegCard[]
  supN: number
  ticketN: number
  noticeN: number
  total: number
  avg: string
}>()

const emit = defineEmits<{ toggle: [flowId: string] }>()

const cardsEl = ref<HTMLElement | null>(null)
// 滚底: 卡片数 + 流式内容变化
useScrollBottom(cardsEl, () => props.segCards.length + '|' + props.segCards.map(c => (c.streamBuffer || '') + (c.reportText || '')).join('\n'))

// 打字机(shownText + shownLen), 完成态切换在 composable 内(card.streaming=false)
const { shownText } = useTypewriter(toRef(props, 'segCards'))

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
          <div class="sc-head" @click="emit('toggle', card.flowId)">
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
          <div v-if="shownText(card, 'think')" class="think-block">
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
