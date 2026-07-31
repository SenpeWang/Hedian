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
watch(() => props.segCards.length, async () => {
  await nextTick()
  if (cardsEl.value) cardsEl.value.scrollTop = cardsEl.value.scrollHeight
})

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
    <div class="panel-title">📋 评价报告</div>
    <div class="panel-body">
      <div class="summary-grid">
        <div><div class="val">{{ supN }}</div><div class="lbl">监护制流程</div></div>
        <div><div class="val">{{ ticketN }}</div><div class="lbl">自唱票流程</div></div>
        <div><div class="val">{{ noticeN }}</div><div class="lbl">信息通报</div></div>
        <div><div class="val">{{ avg }}</div><div class="lbl">平均分</div></div>
        <div><div class="val">{{ total }}</div><div class="lbl">总分</div></div>
      </div>
      <div ref="cardsEl">
        <div v-for="card in segCards" :key="card.flowId"
             class="seg-card" :class="{ collapsed: card.collapsed }"
             :style="{ borderLeftColor: card.streaming ? '#6b7a90' : borderColor(card.flowType) }">
          <div class="sc-head" @click="toggle(card)">
            <span>
              {{ card.streaming ? '🤖' : cardIcon(card.flowType) }}
              {{ card.streaming ? '大模型评估推理中…' : cardLabel(card.flowType) + ' #' + card.flowId + ' [' + card.continueSec + 's]' }}
              <span class="sc-toggle-icon">{{ card.collapsed ? '▶' : '▼' }}</span>
            </span>
            <span v-if="!card.streaming" class="sc-score" :style="{ color: scoreColor(card.score) }">{{ card.score }}/10</span>
          </div>
          <template v-if="!card.streaming">
            <div class="sc-bar"><div class="sc-bar-fill" :style="{ width: card.score * 10 + '%', background: scoreColor(card.score) }"></div></div>
            <div class="sc-detail">{{ card.reportText }}</div>
          </template>
          <template v-else>
            <div class="sc-detail">{{ card.streamBuffer }}</div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>
