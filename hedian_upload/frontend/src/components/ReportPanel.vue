<script setup lang="ts">
import { ref, toRef } from 'vue'
import type { SegCard } from '../types'
import { useScrollBottom } from '../composables/useScrollBottom'
import { useTypewriter } from '../composables/useTypewriter'

const props = defineProps<{
  segCards: SegCard[]
  supN: number
  noticeN: number
  total: number
  avg: string
}>()

// 折叠态回传用卡片主键（flowType:flowId），不能用 flowId——各制度 flow_id 会撞车
const emit = defineEmits<{ toggle: [flowKey: string] }>()

const cardsEl = ref<HTMLElement | null>(null)

// 打字机(shownText + shownLen), 完成态切换在 composable 内(card.streaming=false)
const { shownText } = useTypewriter(toRef(props, 'segCards'))

// 滚底: 依赖打字机已揭示文本(think+report), 使逐字揭示的正文也能实时滚到最新
// 仅依赖 streamBuffer/reportText 会在后端发完、打字机仍逐字揭示时停止滚动, 导致正文追不上
useScrollBottom(cardsEl, () => props.segCards.map(c => shownText(c, 'think') + shownText(c, 'report')).join('\n'))

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
  return '❔'
}
// 卡片头时间标签: 优先显示真实发生区间(2.8s → 133.8s), 让"顺序"看得懂;
// 持续≈0 的流程(读完码同句就收口)显式标"瞬时完成", 避免 0 分看着像流程没跑;
// 拿不到区间(旧版后端)时退回"时长 xx s", 不再只显示一个裸秒数
function timeLabel(card: SegCard): string {
  const s = Number(card.startSec)
  const e = Number(card.endSec)
  const hasRange =
    card.startSec !== null && card.startSec !== undefined && Number.isFinite(s) &&
    card.endSec !== null && card.endSec !== undefined && Number.isFinite(e)
  if (hasRange) {
    if (e - s < 0.5) return `[${s.toFixed(1)}s 瞬时完成]`
    return `[${s.toFixed(1)}s → ${e.toFixed(1)}s]`
  }
  return `[时长 ${card.continueSec}s]`
}
function cardLabel(flowType: string) {
  if (flowType === 'supervision') return '监护制'
  if (flowType === 'info_notice') return '信息通报'
  // 流式首帧类型未到时保持中性文案，避免误标
  return '流程评估'
}
</script>

<template>
  <div class="panel">
    <div class="panel-title">📋 流程评价</div>
    <div class="panel-body" ref="cardsEl">
      <div class="summary-grid">
        <div><div class="val">{{ supN }}</div><div class="lbl">监护制流程</div></div>
        <div><div class="val">{{ noticeN }}</div><div class="lbl">信息通报</div></div>
        <div><div class="val">{{ avg }}</div><div class="lbl">平均分</div></div>
        <div><div class="val">{{ total }}</div><div class="lbl">总分</div></div>
      </div>
      <div>
        <div v-for="card in segCards" :key="card.flowKey"
             class="seg-card" :class="{ collapsed: card.collapsed }"
             :style="{ borderLeftColor: card.streaming ? '#6b7a90' : borderColor(card.flowType) }">
          <div class="sc-head" @click="emit('toggle', card.flowKey)">
            <span>
              {{ card.streaming ? '🤖' : cardIcon(card.flowType) }}
              {{ cardLabel(card.flowType) }} #{{ card.flowId }}{{ card.streaming ? '' : ' ' + timeLabel(card) }}
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
