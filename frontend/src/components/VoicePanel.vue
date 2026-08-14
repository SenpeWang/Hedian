<script setup lang="ts">
import { nextTick, watch, ref } from 'vue'
import type { VoiceEntry } from '../types'

const props = defineProps<{
  entries: VoiceEntry[]
  fmt: (s: number) => string
}>()

const scrollEl = ref<HTMLElement | null>(null)

// 深度监听对话条目（新条目追加或流式 ASR 更新文本），自动平滑滚动到底部最新转录
watch(
  () => [props.entries.length, props.entries.map(e => e.text).join('')],
  async () => {
    await nextTick()
    if (scrollEl.value) {
      scrollEl.value.scrollTo({
        top: scrollEl.value.scrollHeight,
        behavior: 'smooth',
      })
    }
  },
  { deep: true, immediate: true }
)

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!))
}

function renderText(entry: VoiceEntry) {
  if (!entry.text) return ''
  if (!entry.keys || !entry.keys.length) return escapeHtml(entry.text)
  let ranges: [number, number][] = []
  for (const h of entry.keys) {
    let idx = entry.text.indexOf(h)
    while (idx !== -1) {
      ranges.push([idx, idx + h.length])
      idx = entry.text.indexOf(h, idx + 1)
    }
  }
  if (!ranges.length) return escapeHtml(entry.text)
  ranges.sort((a, b) => a[0] - b[0] || (b[1] - b[0]) - (a[1] - a[0]))
  let out = '', last = 0
  for (const r of ranges) {
    if (r[0] < last) continue
    out += escapeHtml(entry.text.slice(last, r[0])) + `<span class="kw-red">${escapeHtml(entry.text.slice(r[0], r[1]))}</span>`
    last = r[1]
  }
  out += escapeHtml(entry.text.slice(last))
  return out
}
</script>

<template>
  <div class="panel">
    <div class="panel-title">
      <span>🎤 人员对话记录</span>
    </div>
    <div class="panel-body voice-body" ref="scrollEl">
      <div
        v-for="(entry, index) in entries"
        :key="entry.sec"
        class="text-item voice-item"
        :class="{ 'latest-voice-item': index === entries.length - 1 }"
      >
        <span class="ts">[{{ fmt(entry.sec) }}]</span>
        <span class="voice-text" v-html="renderText(entry)"></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.voice-body {
  display: flex;
  flex-direction: column;
  gap: 3px;
  scroll-behavior: smooth;
}
.voice-item {
  display: flex;
  align-items: flex-start;
  gap: 4px;
  padding: 3px 6px;
  border-radius: 4px;
  background: rgba(10, 14, 26, 0.6);
  border-left: 2px solid transparent;
  transition: all 0.2s ease;
}
.voice-item:hover {
  background: rgba(30, 42, 66, 0.5);
}
.latest-voice-item {
  border-left: 2px solid #00d4ff;
  background: rgba(0, 212, 255, 0.05);
}
.voice-text {
  flex: 1;
  color: #e0e6f0;
  word-break: break-word;
}
</style>
