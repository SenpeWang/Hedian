<script setup lang="ts">
/**
 * HeaderBar: 顶部标题栏 + 开始/停止控制按钮 + 推理进度条.
 *
 * 通过 start / stop 事件上抛控制意图, 状态与进度由父组件下发.
 */
defineProps<{
  status: 'idle' | 'starting' | 'running' | 'done' | 'stopped'
  progress: number
}>()

const emit = defineEmits<{ start: []; stop: [] }>()
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
      <div v-show="status === 'running' || status === 'starting'" class="progress-group">
        <div class="progress-row">
          <span class="plabel">推理进度</span>
          <div class="pbar">
            <div class="pfill" :style="{ width: (progress || 0) + '%' }" />
          </div>
          <span class="pct">{{ (progress || 0).toFixed(1) }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.progress-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}
.plabel {
  color: #aabbcc;
  white-space: nowrap;
}
.pbar {
  width: 180px;
  height: 12px;
  background: #1a2a3a;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #2a3a4a;
}
.pfill {
  height: 100%;
  background: linear-gradient(90deg, #00d4ff, #00ffcc);
  transition: width 0.3s ease;
  border-radius: 6px;
}
.pct {
  color: #00ffcc;
  min-width: 52px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
</style>
