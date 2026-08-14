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

// 当有新的流程开始/结束系统通知时，自动平滑滚动到底部
watch(
  () => props.flowEvents.length,
  async () => {
    await nextTick()
    if (flowEl.value) {
      flowEl.value.scrollTo({
        top: flowEl.value.scrollHeight,
        behavior: 'smooth',
      })
    }
  },
  { deep: true, immediate: true }
)

function gazeColor() {
  if (!props.gaze.hasHeads) return '#6b7a90'
  return props.gaze.anyInRoi ? '#00ff88' : '#ffaa00'
}

function gazeLabel() {
  if (!props.gaze.hasHeads) return '无人'
  return props.gaze.anyInRoi ? '人注视盘台' : '人未看盘台'
}

function awayPct() {
  return Math.min(100, (props.gaze.awayDuration / 60) * 100).toFixed(0)
}
</script>

<template>
  <div class="panel">
    <div class="panel-title">
      <span>📊 系统通知</span>
    </div>
    <!-- 监控室实时人员与凝视状态看板 -->
    <div class="status-stats">
      <div class="stat-grid">
        <div class="stat-block">
          <div class="stat-block-label">👥 监控室人数</div>
          <div class="stat-block-content">
            <span class="stat-val">{{ people.count }}</span>
            <span class="stat-unit">人</span>
          </div>
          <div class="stat-alert" :style="{ color: people.alertColor }">{{ people.alert }}</div>
        </div>
        <div class="stat-block">
          <div class="stat-block-label">👁️ 凝视状态</div>
          <div class="stat-block-content">
            <span class="stat-val" :style="{ color: gazeColor(), fontSize: gaze.hasHeads ? '22px' : '15px' }">
              {{ gaze.hasHeads ? gaze.headsCount : '待检测' }}
            </span>
            <span class="stat-unit" v-if="gaze.hasHeads">{{ gazeLabel() }}</span>
          </div>
          <div class="gaze-progress-wrap" v-if="gaze.hasHeads && !gaze.anyInRoi">
            <div class="gaze-progress-track">
              <div class="gaze-progress-bar" :style="{
                width: awayPct() + '%',
                background: gaze.awayDuration >= 60 ? '#ff4d4d' : '#ffaa00'
              }"></div>
            </div>
            <span class="gaze-progress-text" :style="{ color: gaze.awayDuration >= 60 ? '#ff4d4d' : '#8899aa' }">
              {{ gaze.awayDuration >= 60 ? '⚠️ ' : '' }}{{ Math.round(gaze.awayDuration) }}/60S
            </span>
          </div>
        </div>
      </div>
    </div>
    <!-- 制度流程起止通知列表（无任何符号，纯文字如：监护制开始 / 监护制结束） -->
    <div class="flow-events" ref="flowEl">
      <div
        v-for="(ev, i) in flowEvents"
        :key="i"
        class="flow-event"
        :style="{ borderLeftColor: ev.color }"
      >
        <span class="ts">[{{ fmt(ev.sec) }}]</span>
        <span class="flow-name" :style="{ color: ev.color }">
          {{ ev.name }}{{ ev.isStart ? '开始' : '结束' }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.flow-events {
  flex: 1;
  overflow-y: auto;
  padding: 6px 8px;
  border-top: 1px solid #1e2a42;
  margin-top: 4px;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.flow-event {
  display: flex;
  align-items: center;
  font-size: 14px;
  padding: 6px 10px;
  border-left: 3px solid #00d4ff;
  background: rgba(10, 14, 26, 0.7);
  border-radius: 4px;
  color: #e0e6f0;
  animation: fadeIn 0.3s;
}
.flow-event .ts {
  color: #6b7a90;
  margin-right: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}
.flow-name {
  font-weight: 600;
  font-size: 14px;
}
</style>
