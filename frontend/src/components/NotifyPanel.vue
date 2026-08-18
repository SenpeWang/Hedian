<!-- 系统通知栏: 两类状态栏
     · 状态量状态栏(监控室人数/凝视状态): getLatestAt 取最新可得值, 持续显示
     · 事件流状态栏(流程事件列表): filter 累积已发生事件, 只增不减 -->
<script setup lang="ts">
import { nextTick, watch, ref } from 'vue'
import type { PeopleState, GazeState, FlowEvent } from '../types'

const props = defineProps<{
  people: PeopleState
  gaze: GazeState
  flowEvents: FlowEvent[]
  fmt: (s: number) => string
}>()

const flowEl = ref<HTMLElement | null>(null)
watch(() => props.flowEvents.length, async () => {
  await nextTick()
  if (flowEl.value) flowEl.value.scrollTop = flowEl.value.scrollHeight
})

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
    <!-- [事件流状态栏] 流程事件列表: 累积显示已发生的流程开始/结束 -->
    <div class="flow-events" ref="flowEl">
      <div v-for="(ev, i) in flowEvents" :key="i" class="flow-event" :style="{ borderLeftColor: ev.color }">
        <span class="ts">[{{ fmt(ev.sec) }}]</span>
        <span :style="{ color: ev.color }">{{ ev.name }}{{ ev.isStart ? '开始' : '结束' }}</span>
      </div>
    </div>
  </div>
</template>
