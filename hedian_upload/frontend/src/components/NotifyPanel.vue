<!-- 系统通知栏: 两类状态栏
     · 状态量状态栏(监控室人数/凝视状态): getLatestAt 取最新可得值, 持续显示
     · 事件流状态栏(流程事件列表): filter 累积已发生事件, 只增不减 -->
<script setup lang="ts">
import { ref } from 'vue'
import type { PeopleState, GazeState, FlowEvent } from '../types'
import { useScrollBottom } from '../composables/useScrollBottom'

const props = defineProps<{
  people: PeopleState
  gaze: GazeState
  flowEvents: FlowEvent[]
  fmt: (s: number) => string
}>()

const flowEl = ref<HTMLElement | null>(null)
useScrollBottom(flowEl, () => props.flowEvents.length)

// 同类流程序号标记: ① ② …（超过 9 个用 #n）。
// 一个视频里可能出现多个同类流程（如中间、结尾各一次"信息通报"），
// 逐行带上序号 + 所属制度名，才能看出这一行判定属于哪一个流程。
const flowTypeName: Record<string, string> = {
  supervision: '监护制',
  info_notice: '信息通报',
  unknown: '流程',
}
const SEQ_MARKS = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']
function seqMark(seq: number): string {
  return seq >= 1 && seq <= SEQ_MARKS.length ? SEQ_MARKS[seq - 1] : `#${seq}`
}
function flowTitle(ev: FlowEvent): string {
  return `${flowTypeName[ev.flowType] || ev.flowType} · 第 ${ev.flowSeq} 个流程`
}

// 监控室人数颜色：>=3人绿色，1-2人黄色，0人红色
function peopleColor(): string {
  const n = typeof props.people.count === 'number' ? props.people.count : parseInt(String(props.people.count), 10)
  if (isNaN(n) || n === 0) return '#ff4d4d'
  if (n >= 3) return '#00ff88'
  return '#ffaa00'
}

// 专注人数计算
function gazeFocusCount(): number | string {
  if (!props.gaze.hasHeads) return '--'
  return props.gaze.anyInRoi ? props.gaze.headsCount : 0
}

// 专注状态颜色：>=2人绿色，1人黄色，0人红色
function gazeFocusColor(): string {
  const val = gazeFocusCount()
  if (val === '--') return '#6b7a90'
  const n = Number(val)
  if (n >= 2) return '#00ff88'
  if (n === 1) return '#ffaa00'
  return '#ff4d4d'
}

function awayPct() {
  return Math.min(100, (props.gaze.awayDuration / 60) * 100).toFixed(0)
}
</script>

<template>
  <div class="panel">
    <div class="panel-title">📊 系统通知</div>
    <!-- [状态量状态栏] 监控室人数 + 凝视状态: 持续显示当前时刻最新值 -->
    <div class="status-stats">
      <div class="stat-grid">
        <div class="stat-block">
          <div class="stat-block-label">👥 监控室</div>
          <div class="stat-block-content">
            <span class="stat-val" :style="{ color: peopleColor() }">{{ people.count }}</span>
            <span class="stat-unit">人</span>
          </div>
        </div>
        <div class="stat-block">
          <div class="stat-block-label">👁️ 凝视状态</div>
          <div class="stat-block-content">
            <span class="stat-val" :style="{ color: gazeFocusColor() }">
              {{ gazeFocusCount() }}
            </span>
            <span class="stat-unit">人专注</span>
          </div>
          <div class="gaze-progress-wrap" v-if="gaze.hasHeads && !gaze.anyInRoi">
            <div class="gaze-progress-track">
              <div class="gaze-progress-bar" :style="{
                width: awayPct() + '%',
                background: gaze.awayDuration >= 60 ? '#ff4d4d' : '#ffaa00'
              }"></div>
            </div>
            <span class="gaze-progress-text" :style="{ color: gaze.awayDuration >= 60 ? '#ff4d4d' : '#94a3b8' }">
              {{ gaze.awayDuration >= 60 ? '⚠️ ' : '' }}{{ Math.round(gaze.awayDuration) }}/60S
            </span>
          </div>
        </div>
      </div>
    </div>
    <!-- [事件流状态栏] 流程事件列表: 流程开始/结束 + 制度层实时下发的子步骤判定 -->
    <div class="flow-events" ref="flowEl">
      <div
        v-for="ev in flowEvents"
        :key="ev.flowType + ev.flowSeq + ev.sec + ev.text"
        class="flow-event"
        :class="{ 'flow-event--step': ev.isStep, ok: ev.ok === true, bad: ev.ok === false }"
        :style="{ borderLeftColor: ev.color }"
      >
        <span class="ts">[{{ fmt(ev.sec) }}]</span>
        <!-- 流程归属标记: 步骤行带制度名（信息通报①），开始/结束行只带序号（文案已含制度名） -->
        <span v-if="ev.flowSeq" class="flow-tag" :style="{ color: ev.color }" :title="flowTitle(ev)">
          {{ (ev.isStep ? (flowTypeName[ev.flowType] || '') : '') + seqMark(ev.flowSeq) }}
        </span>
        <span v-if="ev.isStep" class="mark">{{ ev.ok === true ? '✓' : '✗' }}</span>
        <span :style="ev.isStep ? undefined : { color: ev.color }">{{ ev.text }}</span>
      </div>
    </div>
  </div>
</template>
