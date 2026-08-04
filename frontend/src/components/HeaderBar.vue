<script setup lang="ts">
import type { ProgressState } from '../types'

const props = defineProps<{
  status: string
  statusText: string
  progress: ProgressState
}>()

const emit = defineEmits<{ start: []; stop: [] }>()

function bar(pct: number) {
  const bl = 20, f = Math.round(pct / 100 * bl)
  return pct >= 100 ? '█'.repeat(bl) : '█'.repeat(f) + '░'.repeat(bl - f)
}
</script>

<template>
  <div class="header">
    <h1>⚛️ 核电站行为合规检测系统</h1>
    <div class="btn-group">
      <button
        v-if="status !== 'running' && status !== 'starting'"
        class="start-btn"
        @click="emit('start')"
      >
        {{ status === 'done' ? '🔄 重新测试' : '👤 开始测试' }}
      </button>
      <button
        v-if="status === 'running' || status === 'starting'"
        class="start-btn running-btn"
        disabled
      >
        ⏳ 推理中...
      </button>
      <button
        v-if="status === 'running' || status === 'starting'"
        class="start-btn stop-btn"
        @click="emit('stop')"
      >
        🛑 停止测试
      </button>
    </div>
    <div class="header-right">
      <div class="status-dot" :class="{ active: status === 'running' }"></div>
      <span class="header-status">{{ statusText }}</span>
      <div class="progress-group" v-show="status === 'running'">
        <div class="progress-row">🎤 |{{ bar(progress.voice) }}| {{ progress.voice.toFixed(1) }}%</div>
        <div class="progress-row">🎯 |{{ bar(progress.tracker) }}| {{ progress.tracker.toFixed(1) }}%</div>
        <div class="progress-row">👁️ |{{ bar(progress.gaze) }}| {{ progress.gaze.toFixed(1) }}%</div>
      </div>
    </div>
  </div>
</template>
